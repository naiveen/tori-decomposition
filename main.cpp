
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/edge_topology.h>
#include <igl/heat_geodesics.h>
#include <igl/barycenter.h>
#include <igl/cut_mesh.h>
#include <igl/adjacency_matrix.h>
#include <fstream>
#include <vector>
#include <forward_list>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"
#include <stack>
#include <math.h>

#define  DEBUG 0
#define  primalSpanViewFlag 0
#define  dualSpanViewFlag 0
#define  FERTILITY 0
#define  VISUALIZEHANDLES 1
#define  CUTMESH 1

igl::opengl::glfw::Viewer viewer;
igl::HeatGeodesicsData<double> data;
Eigen::MatrixXd PD1, PD2;
Eigen::VectorXd PV1, PV2;

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::SparseMatrix<double> VF;
Eigen::MatrixXi EV; //Edges defined by vertex
Eigen::MatrixXi FE; //Face to edge
Eigen::MatrixXi EF; //Also dual graph edge vertices in FCV
Eigen::MatrixXd FCV; // Face centers, Dual edge vertices
Eigen::MatrixXd DEW; // Dual edge weights
Eigen::MatrixXd EVW; // Primary edge weights

std::vector<std::vector<int>> goodCycles;
std::vector<std::vector<int>> tunnelCycles;
std::unordered_set<int> goodEdgeIndexVec;
std::unordered_set<int> badEdgeIndexVec;
std::vector<std::unordered_map<int, int>> VVE;
std::vector<std::unordered_map<int, int>> dVVE;

const Eigen::RowVector3d red(0.8, 0.2, 0.2), blue(0.2, 0.2, 0.8), green(0.2, 0.8, 0.2), color1(0.2, 0.2, 0.2);

// used for the criteria of determining a good cycle
bool CompareCycles(std::vector<int>&, std::vector<int>&);
float DistToCentroid(std::vector<int>&);

struct WeightedEdge {
    float dist = 0, _dist = 0;
    int src, dst, index;
    WeightedEdge(int index, int src, int dst, float dist) : index(index), src(src), dst(dst), dist(dist), _dist(dist) {};
};
struct RingStrip {
    int center, node1, node2;
    RingStrip(int center, int node1, int node2) : center(center), node1(node1), node2(node2) {};
};
struct Edge {
    int src, dst;
    Edge(int src, int dst) : src(src), dst(dst) {};
};



class UnionFind {
private:
    std::vector<int> root_, size_;
public:
    UnionFind(int n) {
        root_.resize(n);
        size_.resize(n, 1);
        // init each node is it's own root
        for (int i = 0; i < n; i++)
            root_[i] = i;
    }

    int getRoot(int v) {
        while (root_[v] != v) {
            root_[v] = root_[root_[v]];
            v = root_[v];
        }
        return root_[v];
    }

    void Union(int a, int b) {
        int root_a = getRoot(a);
        int root_b = getRoot(b);
        // make the smaller subtree part of bigger
        if (size_[root_a] > size_[root_b]) {
            root_[root_b] = root_a;
            size_[root_a] += size_[root_b];
        }
        else {
            root_[root_a] = root_b;
            size_[root_b] += size_[root_a];
        }
    }
    // check if vertex a and b are connected
    bool Find(int a, int b) {
        int root_a = getRoot(a);
        int root_b = getRoot(b);
        return root_a == root_b;
    }
};

void kruskalMST(int n, std::vector<WeightedEdge>& edges, std::vector<WeightedEdge>& span) {
    int cost = 0;
    // sort the edges by weight
    sort(edges.begin(), edges.end(), [](WeightedEdge& a, WeightedEdge& b) -> bool {
        return a.dist < b.dist;
        });

    UnionFind ob(n);

    for (auto edge : edges) {
        int src = edge.src, dst = edge.dst;
        if (!ob.Find(src, dst)) {
            ob.Union(src, dst);
            span.push_back(edge);
        }
    }
    sort(edges.begin(), edges.end(), [](WeightedEdge& a, WeightedEdge& b) -> bool {
        return a.index < b.index;
        });
}


void visualize_cycle(igl::opengl::glfw::Viewer& viewer, std::vector<int>& cycle, Eigen::RowVector3d color) {
    /*
    * Visualize one cycle
    * cycle contains a list of vertex indices that represent the cycle
    */
    viewer.data().add_edges(V.row(cycle[0]), V.row(cycle.back()), green);
    viewer.data().add_edges(V.row(cycle[0]), V.row(cycle[1]), color1);
    for (auto j = 1; j < cycle.size(); j++) {
        viewer.data().add_edges(V.row(cycle[j - 1]), V.row(cycle[j]), color);
    }
}

void visualize_cycles(igl::opengl::glfw::Viewer& viewer, std::vector<std::vector<int>>& cycles, Eigen::RowVector3d color) {
    int i = 0;
    sort(cycles.begin(), cycles.end(), [](std::vector<int>& a, std::vector<int>& b) -> bool {
        return a.size() < b.size();
        });
    for (auto cycle : cycles) {
        //i++;
        if (i > 3)
            continue;
        visualize_cycle(viewer, cycle, color);
    }
}

void dfs(std::vector<std::vector<int>>& VE, std::vector<int>& cycle, int prevIndex, int curIndex, bool& cycleflag) {

    if (curIndex == cycle[0]) {
        cycleflag = true;
        return;
    }
    for (auto i = 0; i < VE[curIndex].size(); i++) {
        if (VE[curIndex][i] == prevIndex)
            continue;
        dfs(VE, cycle, curIndex, VE[curIndex][i], cycleflag);
        if (cycleflag) {
            cycle.push_back(curIndex);
            break;
        }
    }
    return;
}


void bridge_dfs(int vertex,int parent, int &counter, std::unordered_set<int>& uos, std::unordered_set<int>& bridge_uos, std::vector<int>& dfs_numbers, std::vector<int>& escape_numbers) {
    if (dfs_numbers[vertex] != -1) {
        return;
    }
    int flag = 0;
    for (auto i = 0; i < 3; i++) {
        flag += uos.count(FE(vertex, i)) == 1;
    }
    if (flag==3) return;
    dfs_numbers[vertex] = counter++;
    escape_numbers[vertex] = dfs_numbers[vertex];
    for (auto i = 0; i < 3; i++) {
        if (uos.count(FE(vertex, i)) == 1) {
            continue;
        }
        int next_vertex = EF(FE(vertex, i), 0)!= vertex ? EF(FE(vertex, i), 0) : EF(FE(vertex, i), 1);
        if (dfs_numbers[next_vertex] == -1) {
            bridge_dfs(next_vertex, vertex, counter, uos, bridge_uos, dfs_numbers, escape_numbers);
            escape_numbers[vertex] = std::min(escape_numbers[vertex], escape_numbers[next_vertex]);
            
            if (escape_numbers[next_vertex] >  dfs_numbers[vertex]) {
                bridge_uos.insert(FE(vertex, i));
                //viewer.data().add_edges(FCV.row(vertex), (V.row(EV(FE(vertex, i), 0)) + V.row(EV(FE(vertex, i), 1))) / 2, blue);
                //viewer.data().add_edges(FCV.row(next_vertex), (V.row(EV(FE(vertex, i), 0)) + V.row(EV(FE(vertex, i), 1))) / 2, blue);
            }
        }
        else if (next_vertex != parent) {
            escape_numbers[vertex] = std::min(escape_numbers[vertex], dfs_numbers[next_vertex]);
        }
    }
}


bool get_cycle_diff(std::vector<int>& cycle1, std::vector<int>& cycle2) {
    if (cycle1.size() != cycle2.size()) {
        return 1;
    }
    for (auto i = 0; i < cycle1.size(); i++) {
        if (cycle1[i] != cycle2[i]) {
            return 1;
        }
    }
    return 0;
}

float get_tight_cycle_metric(int v1, int v2) {
    return   (V.row(v1) - V.row(v2)).norm();
}

void init_dist_map(std::vector<int>& cycle, std::unordered_map<int, std::pair<int, float>>& dist_map) {
    for (auto i = 1; i < cycle.size(); i++) {
        dist_map[cycle[i]].first = cycle[i - 1];
        dist_map[cycle[i]].second = dist_map[cycle[i-1]].second + get_tight_cycle_metric(cycle[i - 1], cycle[i]);
    }
}

float get_cycle_metric(std::vector<int>& cycle) {
    float dist_metric = 0;
    for (auto i = 1; i < cycle.size(); i++) {
        dist_metric += get_tight_cycle_metric(cycle[i - 1], cycle[i]);
    }
    dist_metric += get_tight_cycle_metric(cycle.front(), cycle.back());
    return dist_metric;
}


void remove_if_dangling_edge(int edge, std::unordered_set<int>& uos) {
    if (uos.count(edge) == 1) {
        return;
    }
    if (uos.count(FE(EF(edge, 0), 0)) + uos.count(FE(EF(edge, 0), 1)) + uos.count(FE(EF(edge, 0), 2)) == 2) {
        uos.insert(edge);
        if (uos.count(FE(EF(edge, 1), 0)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 1), 0), uos);
        if (uos.count(FE(EF(edge, 1), 1)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 1), 1), uos);
        if (uos.count(FE(EF(edge, 1), 2)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 1), 2), uos);
    }
    if (uos.count(FE(EF(edge, 1), 0)) + uos.count(FE(EF(edge, 1), 1)) + uos.count(FE(EF(edge, 1), 2)) == 2) {
        uos.insert(edge);
        if (uos.count(FE(EF(edge, 0), 0)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 0), 0), uos);
        if (uos.count(FE(EF(edge, 0), 1)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 0), 1), uos);
        if (uos.count(FE(EF(edge, 0), 2)) == 0)
            remove_if_dangling_edge(FE(EF(edge, 0), 2), uos);
    }
}


void create_bridge_graph(std::vector<WeightedEdge>& dualEdges, std::unordered_set<int>& uos) {
    std::vector<std::unordered_map<int, int>> bridgeVVE;
    for (auto edge : dualEdges) {
        for (auto it = VVE[edge.src].begin(); it != VVE[edge.src].end(); it++) {
            if(uos.count(it->second)==0)
                bridgeVVE[edge.index][it->second] = 1;
        }
        for (auto it = VVE[edge.dst].begin(); it != VVE[edge.dst].end(); it++) {
            if (uos.count(it->second) == 0)
                bridgeVVE[edge.index][it->second] = 1;
        }
    }
}




void tree_cotree(std::vector<WeightedEdge>& primalEdges, std::vector<WeightedEdge>& dualEdges, std::vector<std::vector<int>>& cycles, igl::opengl::glfw::Viewer& viewer) {
    /*
    * Given Primary and Dual edges, the function computes 2*g good cycles.
    */
    std::vector<WeightedEdge>  primalSpan;
    std::vector<WeightedEdge> dualEdges_;
    std::vector<WeightedEdge> dualSpan;
    std::vector<std::vector<int>>VE(V.rows(), std::vector<int>());
    std::unordered_set<int> uos;
    std::unordered_set<int> bridge_uos;

    //Assign low weight to edges that form good cycles
    for (auto& index : goodEdgeIndexVec) {
        primalEdges[index].dist = 0;
        //dualEdges[index].dist = INT_MAX;
    }
    for (auto& index : badEdgeIndexVec) {
        primalEdges[index].dist = INT_MAX;
        uos.insert(index);
        //dualEdges[index].dist = INT_MAX;
    }
    kruskalMST(V.rows(), primalEdges, primalSpan);

   
    //Add vertex to edge relation for edges in spanning tree
    for (auto edge : primalSpan) {
        uos.insert(edge.index);
        VE[edge.dst].push_back(edge.src);
        VE[edge.src].push_back(edge.dst);

        if (primalSpanViewFlag) {
            viewer.data().add_edges(V.row(edge.src), V.row(edge.dst), green);
        }

    }

    //Assign high weight to dual graph edges corresponding to primary graph edges
    
    for (auto& index : uos) {
        dualEdges[index].dist = INT_MAX;
    }
    //Remove Dangling edges
    for (auto edge : dualEdges) {
        remove_if_dangling_edge(edge.index, uos);
    }

    if (dualSpanViewFlag) {
        for (auto edge : dualEdges) {
            if (uos.count(edge.index) == 0) {
                //viewer.data().add_edges(V.row(EV(edge.index, 0)), V.row(EV(edge.index, 1)), blue);
                viewer.data().add_edges(FCV.row(edge.src), (V.row(EV(edge.index, 0)) + V.row(EV(edge.index, 1))) / 2, red);
                viewer.data().add_edges(FCV.row(edge.dst), (V.row(EV(edge.index, 0)) + V.row(EV(edge.index, 1))) / 2, red);
            }
        }
    }

    //Remove Bridge edges
    int bridge_counter = 0;
    std::vector<int>dfs_numbers(FCV.rows(),-1);
    std::vector<int>escape_numbers(FCV.rows(),-1);
    for (auto i = 0; i < FCV.rows(); i++) {
        bridge_dfs(i, i, bridge_counter, uos, bridge_uos, dfs_numbers, escape_numbers);
    }

    for (auto &edge : bridge_uos) {
        uos.insert(edge);
    }
    /*
    for (auto edge : dualEdges) {
        if (uos.count(edge.index)==0) {
            dualEdges_.push_back(edge);
        }
    }
    */

    //Spanning tree computation on dual Graph
    //kruskalMST(FCV.rows(), dualEdges_, dualSpan);
    //Reset dual graph edge weights back
    for (auto& index : uos) {
        dualEdges[index].dist = dualEdges[index]._dist;
    }
    for (auto& index : goodEdgeIndexVec) {
        primalEdges[index].dist = primalEdges[index]._dist;
        dualEdges[index].dist = dualEdges[index]._dist;
    }
    for (auto i = 0; i < EV.rows(); i++) {
        if (uos.count(i) == 0) {
            std::vector<int> cycle;
            cycle.push_back(EV(i, 0));
            cycle.push_back(EV(i, 1));
            bool cycleFlag = 0;
            dfs(VE, cycle, cycle[0], cycle[1], cycleFlag);
            cycle.push_back(cycle[0]);
            reverse(cycle.begin(), cycle.end());
            cycle.pop_back();
            cycle.pop_back();
            cycles.push_back(cycle);
        }
    }
}


void get_start_strip(std::vector<int>& cycle, std::vector<int>& cycle1, std::vector<int>& cycle2, std::unordered_set<int>& cycle_edges, 
    std::unordered_set<int>& node_set, std::unordered_set<int>& start_strip, igl::opengl::glfw::Viewer& viewer) {
    int k = 0;
    while (1){
        int node, node1 = 0, node2 = 0;
        int startNode = cycle[k];
        int nextNode = cycle[++k];
        int edge = VVE[startNode][nextNode];
        int f1 = EF(edge, 0);
        int f2 = EF(edge, 1);
        for (auto i = 0; i < F.row(f1).size(); i++) {
            node = F(f1, i);
            if (node != startNode && node != nextNode) {
                node1 = node;
            }
        }
        for (auto i = 0; i < F.row(f2).size(); i++) {
            node = F(f2, i);
            if (node != startNode && node != nextNode) {
                node2 = node;
            }
        }
        if (node_set.count(node1) != 0 && node_set.count(node2) != 0) {
            start_strip.insert(startNode);
            start_strip.insert(node1);
            start_strip.insert(node2);
            cycle1.push_back(node1);
            cycle2.push_back(node2);
            node_set.erase(node1);
            node_set.erase(node2);
            break;
        }
    }
    int index = k;
    while (cycle1.size()==1) {
        index %= cycle.size();
        for (auto it = VVE[cycle[index]].begin(); it != VVE[cycle[index]].end(); it++) {
            if (node_set.count(it->first)!=0 && VVE[cycle1[0]].count(it->first) != 0) {
                cycle1.push_back(it->first);
                node_set.erase(it->first);
                break;
            }
        }
        index++;
    }
    index = k;
    while (cycle2.size() == 1) {
        index %= cycle.size();
        for (auto it = VVE[cycle[index]].begin(); it != VVE[cycle[index]].end(); it++) {
            if (node_set.count(it->first) != 0 && VVE[cycle2[0]].count(it->first) != 0) {
                cycle2.push_back(it->first);
                node_set.erase(it->first);
                break;
            }
        }
        index++;
    }

    
    do{
        if (DEBUG) {
            visualize_cycle(viewer, cycle, blue);
            viewer.data().add_edges(V.row(cycle1[0]), V.row(cycle1[1]), blue);
            viewer.data().add_edges(V.row(cycle2[0]), V.row(cycle2[1]), blue);
            for (auto edge : cycle_edges)
                viewer.data().add_edges(V.row(EV(edge, 0)), V.row(EV(edge, 1)), red);
            return;
        }
        for (auto it = VVE[cycle1.back()].begin(); it != VVE[cycle1.back()].end(); it++) {
            if (node_set.count(it->first) != 0 && cycle_edges.count(it->second) != 0) {
                cycle1.push_back(it->first);
                node_set.erase(it->first);
                break;
            }
        }
    }while (VVE[cycle1.front()].count(cycle1.back()) == 0);

    do{
        for (auto it = VVE[cycle2.back()].begin(); it != VVE[cycle2.back()].end(); it++) {
            if (node_set.count(it->first) != 0 && cycle_edges.count(it->second) != 0) {
                cycle2.push_back(it->first);
                node_set.erase(it->first);
                break;
            }
        }
    }while (VVE[cycle2.front()].count(cycle2.back()) == 0);
    k--;
    int initialCycleSize = cycle.size();
    for (auto i = 0; i < k; i++) {
        cycle.push_back(cycle[i]);
    }
    reverse(cycle.begin(), cycle.end());
    while (cycle.size() != initialCycleSize) {
        cycle.pop_back();
    }
    reverse(cycle.begin(), cycle.end());
}


void remove_loops(std::list<int>& fl) {
    std::unordered_set<int> uos;
    std::stack<int> s;
    for (auto it = fl.begin(); it != fl.end(); it++) {
        if (uos.count(*it) != 0) {
            while (s.top() != *it) {
                uos.erase(s.top());
                s.pop();
            }
        }
        else {
            s.push(*it);
            uos.insert(*it);
        }
    }
    fl.clear();
    while (s.size() != 0) {
        fl.push_front(s.top());
        s.pop();
    }
}
void edge_tight_cycle(std::vector<int>& cycle) {
    std::list<int> fl;
    int count = cycle.size();

    for (auto i = 0; i < cycle.size(); i++) {
        fl.push_front(cycle[i]);
    }
    auto it1 = fl.begin();
    auto it2 = fl.begin();
    auto it3 = fl.begin();
    it2++;
    it3++;
    it3++;
    bool flag = 1;
    bool check = 1;

    while (flag) {
        if (VVE[*it1].count(*it3)) {
            fl.remove(*it2);
            check = 0;
        }
        else {
            for (auto node : VVE[*it1]) {
                if (VVE[*it2].count(node.first) && VVE[*it3].count(node.first)) {
                    //*it2 = node.first;
                    break;
                }
            }

            it1++;
        }
        it2 = it3;
        it3++;

        if (it2 == fl.end()) {
            it2 = fl.begin();
        }
        if (it3 == fl.end()) {
            it3 = fl.begin();
        }

        if (it1 == fl.end()) {
            if (check)
                flag = 0;
            check = 1;
            if (fl.back() == *it2) {
                fl.erase(fl.begin());
                fl.erase(it2);
            }
            //remove_loops(fl);
            it1 = fl.begin();
            
            /*
            if (DEBUG) {
                visualize_cycle(viewer, cycle, blue);
                cycle.clear();
                for (auto it : fl) {
                    cycle.push_back(it);
                }
                visualize_cycle(viewer, cycle, red);
                flag = 0;
                DEBUG = 0;
            }
            */
        }
    }
    cycle.clear();
    for (auto it : fl) {
        cycle.push_back(it);
    }
}

/*
void edge_tight_cycle(std::vector<int>& cycle) {
    std::list<int> fl;
    int index = 0;

    for (auto i = 0; i < cycle.size(); i++) {

        fl.push_front(cycle[i]);
    }
    fl.push_front(cycle[0]);
    fl.push_front(cycle[1]);

    bool flag = 0;
    auto it1 = fl.begin();
    auto it2 = fl.begin();
    it2++;
    it2++;
    while (it2 != fl.end()) {
        if (VVE[*it1].count(*it2)) {
            auto temp = it1;
            it1++;
            fl.remove(*it1);
            it1 = temp;
            if (it1 == fl.begin()) {
                it2++;
            }
            else {
                it1--;
            }
        }
        else {
            it1++;
            it2++;
        }
    }
    cycle.clear();
    for (auto it : fl) {
        if (cycle.size() > 0 && it == cycle[0]) {
            break;
        }
        cycle.push_back(it);

    }
    if (VVE[cycle.back()].count(cycle[1]) != 0) {
        reverse(cycle.begin(), cycle.end());
        cycle.pop_back();
        edge_tight_cycle(cycle);
    }
}
*/

int get_adj_node_count(int node, std::unordered_set<int>& cycle_edges) {
    int count = 0;
    for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {
        count += cycle_edges.count(it1->second);
    }
    return count;
}
void remove_branches(int node, std::unordered_set<int>& cycle_edges, std::unordered_set<int>& node_set) {

    while (1) {
        int count = get_adj_node_count(node, cycle_edges);
        if (count > 1) {
            return;
        }
        if (count == 0) {
            //node_set.erase(node);
            return;
        }
        //node_set.erase(node);
        for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {
            if (cycle_edges.count(it1->second) != 0) {
                cycle_edges.erase(it1->second);
                node = it1->first;
                break;
            }
        }
    }
}
void remove_triangles(int node, std::unordered_set<int>& cycle_edges) {

    while (1) {
        if (get_adj_node_count(node, cycle_edges) != 2) {
            return;
        }
        int node1, node2;
        for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {
            if (cycle_edges.count(it1->second) != 0) {
                node1 = it1->first;
                break;
            }
        }
        for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {
            if (cycle_edges.count(it1->second) != 0 && it1->first != node1) {
                node2 = it1->first;
                break;
            }
        }
        if (VVE[node1].count(node2) != 0 && cycle_edges.count(VVE[node1][node2]) != 0) {
            cycle_edges.erase(VVE[node1][node]);
            cycle_edges.erase(VVE[node2][node]);
            remove_triangles(node1, cycle_edges);
            remove_triangles(node2, cycle_edges);
        }
        else {
            return;
        }
    }

    /*
    while (1) {
        int count = get_adj_node_count(node, cycle_edges);
        if (count == 3) {
            int node1, node2;
            for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {

                    auto temp2 = 0;
                for (auto it2 = VVE[it1->first].begin(); it2 != VVE[it1->first].end(); it2++) {
                    temp2 += cycle_edges.count(it2->second);
                }
                if (temp2 == 2) {
                    for (auto it2 = VVE[it1->first].begin(); it2 != VVE[it1->first].end(); it2++) {
                        cycle_edges.erase(it2->second);
                    }
                    cycle_edges.erase(it1->second);
                }
            }
        }
    }
    */
}
bool check_valid_edge(int edge, std::unordered_set<int>& main_cycle_set) {
    int f1 = EF(edge, 0);
    int f2 = EF(edge, 1);
    int node;
    for (auto i = 0; i < F.row(f1).size(); i++) {
        node = F(f1, i);
        if (main_cycle_set.count(node) != 0)
            return true;
    }
    for (auto i = 0; i < F.row(f2).size(); i++) {
        node = F(f2, i);
        if (main_cycle_set.count(node) != 0)
            return true;
    }
    return false;
}
void one_ring_tight_scheduler(std::vector<int>& cycle, std::vector<int>& newCycle) {
    std::unordered_set<int> node_set;
    std::unordered_set<int> bridge_edges;
    std::unordered_set<int> cycle_edges;
    std::vector<int> cycle1;
    std::vector<int> cycle2;
    std::unordered_set<int> main_cycle_set;
    std::unordered_set<int> cycle1_set;
    std::unordered_set<int> cycle2_set;
    std::unordered_set<int> outer_cycle_set;
    std::queue<int> q;
    std::unordered_set<int> start_strip;
    std::unordered_map<int, std::pair<int, float>> dist_map;

    //Edges that touch the main cycle
    for (auto node : cycle) {
        main_cycle_set.insert(node);
        for (auto it = VVE[node].begin(); it != VVE[node].end(); it++) {
            node_set.insert(it->first);
            bridge_edges.insert(it->second);
            if (DEBUG)
                viewer.data().add_edges(V.row(node), V.row(it->first), blue);
        }
    }

    //Outer cycle edges that run parallel to the main cycle
    for (auto node : cycle) {
        for (auto it1 = VVE[node].begin(); it1 != VVE[node].end(); it1++) {
            for (auto it2 = VVE[it1->first].begin(); it2 != VVE[it1->first].end(); it2++) {
                if (node_set.count(it2->first) != 0 && bridge_edges.count(it2->second) == 0) {
                    // Edges that correspond to outer cycles
                    cycle_edges.insert(it2->second);
                }
            }
        }
    }
    //Remove loops in outer circle
    for (auto node : node_set) {
        if (main_cycle_set.count(node) != 0)
            continue;
        int count = get_adj_node_count(node, cycle_edges);
        if (count < 2)
            remove_branches(node, cycle_edges, node_set);//removes branches in outer cycle edges
    }
    for (auto node : node_set) {
        if (main_cycle_set.count(node) != 0)
            continue;
        int count = get_adj_node_count(node, cycle_edges);
        if (count == 2) {
            remove_triangles(node, cycle_edges); //removes triangles in outer cycle edges
        }
    }
    for (auto node : node_set) {
        if (main_cycle_set.count(node) != 0)
            continue;
        int count = get_adj_node_count(node, cycle_edges);
        if (count < 2)
            remove_branches(node, cycle_edges, node_set);//removes branches in outer cycle edges
    }

    for (auto node : node_set) {
        if (main_cycle_set.count(node) != 0)
            continue;
        int count = get_adj_node_count(node, cycle_edges);
        if (count >2) {
            for (auto it = VVE[node].begin(); it != VVE[node].end(); it++) {
                if (cycle_edges.count(it->second) == 0)
                    continue;
                if (!check_valid_edge(it->second, main_cycle_set))
                    cycle_edges.erase(it->second);
            }
        }
    }


    node_set.clear();
    for (auto edge : cycle_edges) {
        node_set.insert(EV(edge, 0));
        node_set.insert(EV(edge, 1));
    }

    get_start_strip(cycle, cycle1, cycle2, cycle_edges, node_set, start_strip, viewer);
    node_set.clear();
    for (auto node : cycle1) {
        node_set.insert(node);
    }
    for (auto node : cycle2) {
        node_set.insert(node);
    }
    if (DEBUG) {
        return;
    }
    int startIndex = cycle[0];
    startIndex = cycle.front();
    dist_map[cycle.front()] = std::make_pair(startIndex, 0);
    dist_map[cycle1.front()] = std::make_pair(startIndex, get_tight_cycle_metric(startIndex, cycle1.front()));
    dist_map[cycle2.front()] = std::make_pair(startIndex, get_tight_cycle_metric(startIndex, cycle2.front()));

   
    init_dist_map(cycle, dist_map);
    init_dist_map(cycle1, dist_map);
    init_dist_map(cycle2, dist_map);
    for (auto i = 1; i < cycle1.size(); i++) {
        int currNode = cycle1[i];
        for (auto it = VVE[currNode].begin(); it != VVE[currNode].end(); it++) {
            if (i != 1 && start_strip.count(it->first) != 0) {
                continue;
            }
            if (main_cycle_set.count(it->first) != 0) {
                auto temp = get_tight_cycle_metric(it->first, currNode) + dist_map[it->first].second;
                if (temp < dist_map[currNode].second) {
                    dist_map[currNode].first = it->first;
                    dist_map[currNode].second = temp;
                }
            }
        }
    }
    for (auto i = 1; i < cycle2.size(); i++){
        int currNode = cycle2[i];
        for (auto it = VVE[currNode].begin(); it != VVE[currNode].end(); it++) {
            if (i != 1 && start_strip.count(it->first) != 0) {
                continue;
            }
            if (main_cycle_set.count(it->first) != 0) {
                auto temp = get_tight_cycle_metric(it->first, currNode) + dist_map[it->first].second;
                if (temp < dist_map[currNode].second) {
                    dist_map[currNode].first = it->first;
                    dist_map[currNode].second = temp;
                }
            }
        }
    }
    for (auto i = 1; i < cycle.size(); i++) {
        int currNode = cycle[i];
        for (auto it = VVE[currNode].begin(); it != VVE[currNode].end(); it++) {
            if (i != 1 && start_strip.count(it->first) != 0) {
                continue;
            }
            if (node_set.count(it->first) != 0) {
                auto temp = get_tight_cycle_metric(it->first, currNode) + dist_map[it->first].second;
                if (temp < dist_map[currNode].second) {
                    dist_map[currNode].first = it->first;
                    dist_map[currNode].second = temp;
                }
            }
        }
    }

    float cycle_metric0 = dist_map[cycle.back()].second;
    float cycle_metric1 = dist_map[cycle1.back()].second;
    float cycle_metric2 = dist_map[cycle2.back()].second;

    int bestCyclePos = cycle[0];
    float bestMetric = INT_MAX;
    if (dist_map[cycle.back()].second < bestMetric) {
        bestMetric = dist_map[cycle.back()].second;
        bestCyclePos = cycle.back();
    }
    if (dist_map[cycle1.back()].second < bestMetric) {
        bestMetric = dist_map[cycle1.back()].second;
        bestCyclePos = cycle1.back();
    }
    if (dist_map[cycle2.back()].second < bestMetric) {
        bestMetric = dist_map[cycle2.back()].second;
        bestCyclePos = cycle2.back();
    }
    newCycle.push_back(bestCyclePos);
    while (dist_map[newCycle.back()].first != newCycle.back()) {
        newCycle.push_back(dist_map[newCycle.back()].first);
    }
    if (VVE[newCycle.back()].count(newCycle.front()) == 0) {
        if (newCycle.front() == cycle1.back()) {
            newCycle.push_back(cycle1.front());
        }
        else {
            newCycle.push_back(cycle2.front());
        }
    }
    reverse(newCycle.begin(), newCycle.end());

    if (DEBUG) {
        visualize_cycle(viewer, newCycle, green);
        visualize_cycle(viewer, cycle, blue);
        
        visualize_cycle(viewer, cycle1, red);
        visualize_cycle(viewer, cycle2, red);
        return;
    }
}

void tighten_cycle(std::vector<int>& cycle) {
    int count = 0;
    if (DEBUG) {
        visualize_cycle(viewer, cycle, blue);
    }
    edge_tight_cycle(cycle);
    std::vector<int> newCycle;
    if (DEBUG) {
        visualize_cycle(viewer, cycle, red);
        return;
    }
    while (count < 50) {
        newCycle.clear();
        one_ring_tight_scheduler(cycle, newCycle);
        if (DEBUG) {
            return;
        }
        if (get_cycle_diff(cycle, newCycle) == 0) {
            break;
        }
        edge_tight_cycle(newCycle);
        cycle = newCycle;
        count++;
    }
}

void find_good_cycles(std::vector<std::vector<int>>& cycles, igl::opengl::glfw::Viewer& viewer) {
    //Find a good cycle and store it in goodCycles
    std::vector<std::vector<int>> uniqueCycles;
    for (auto cycle : cycles) {
        
        if (DEBUG) {
            return;
        }
        int count = 0;
        if (goodEdgeIndexVec.count(VVE[cycle[0]][cycle.back()]) > 0) {
            count = 1;
        }
        for (auto j = 1; j < cycle.size(); j++) {
            if (goodEdgeIndexVec.count(VVE[cycle[j - 1]][cycle[j]]) > 0) {
                count += 1;
            }
        }
        if (count < cycle.size() / 2) {
            uniqueCycles.push_back(cycle);
        }
    }
    sort(uniqueCycles.begin(), uniqueCycles.end(), [](std::vector<int>& a, std::vector<int>& b) -> bool {
        return a.size() < b.size();
        });

    if (DEBUG) {
        visualize_cycles(viewer, uniqueCycles, blue);
        return;
    }

    if (uniqueCycles.size() > 0) {
        tighten_cycle(uniqueCycles[0]);
        goodCycles.push_back(uniqueCycles[0]);
        //Add edge index of the good cycle to goodEdgeIndexVec
        for (auto j = 1; j < uniqueCycles[0].size(); j++) {
            goodEdgeIndexVec.insert(VVE[uniqueCycles[0][j - 1]][uniqueCycles[0][j]]);
        }
        goodEdgeIndexVec.insert(VVE[uniqueCycles[0][0]][uniqueCycles[0].back()]);
        badEdgeIndexVec.insert(VVE[uniqueCycles[0][0]][uniqueCycles[0].back()]);
    }
}

void get_tunnel_cycles(std::vector<std::vector<int>>& goodCycles, std::vector<std::vector<int>>& tunnelCycles) {
    sort(goodCycles.begin(), goodCycles.end(), [](std::vector<int>& a, std::vector<int>& b) -> bool {
        return a.size() < b.size();
        });

    for (auto i = 0; i < goodCycles.size() / 2; i++) {
        tunnelCycles.push_back(goodCycles[i]);
    }

    if (FERTILITY) {
        tunnelCycles.clear();
        tunnelCycles.push_back(goodCycles[4]);
        tunnelCycles.push_back(goodCycles[5]);
        tunnelCycles.push_back(goodCycles[6]);
        tunnelCycles.push_back(goodCycles[7]);
    }

}

void cut_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::unordered_set<int>& cutEdgeSet) {
    Eigen::MatrixXi cutF;
    Eigen::MatrixXi FF;
    Eigen::MatrixXi nFF;
    FF.resize(F.rows(), 3);
    std::unordered_set<int> newFaceSet;

    std::unordered_set<int> cutVertexSet;
    double x = 0, y = 0, z = 0;

    for (auto edge : cutEdgeSet) {
        cutVertexSet.insert(EV(edge, 0));
        cutVertexSet.insert(EV(edge, 1));
    }
    
    int ffCount = 0;
    for (auto i = 0; i < FE.rows(); i++) {
        if (cutEdgeSet.count(FE(i, 0)) == 0 && cutEdgeSet.count(FE(i, 1)) == 0 && cutEdgeSet.count(FE(i, 2)) == 0) {
            newFaceSet.insert(i);
        }
        else {
            int edgeIndex=0;
            if (cutEdgeSet.count(FE(i, 0)) == 0)
                edgeIndex = FE(i, 0);
            else if (cutEdgeSet.count(FE(i, 1)) == 0)
                edgeIndex = FE(i, 1);
            else if (cutEdgeSet.count(FE(i, 2)) == 0)
                edgeIndex = FE(i, 2);
        }
    }  
    nFF.resize(newFaceSet.size(), 3);
    int nFFCount = 0;
    for (auto i = 0; i < FE.rows(); i++) {
        if (newFaceSet.count(i)) {
            nFF.row(nFFCount++) << F(i, 0), F(i, 1), F(i, 2);
        }
    }

    viewer.data().set_mesh(V, nFF);
}

void get_cut_edges(std::vector<std::vector<int>>& tunnelCycles, std::unordered_set<int> &cutEdgeSet) {
    Eigen::VectorXd DSource,DSink;
    int sourceCount = tunnelCycles.back().size();
    int sinkCount = 0;
    for (auto cycle : tunnelCycles) {
        sinkCount += cycle.size();
    }
    sinkCount -= sourceCount;
    Eigen::VectorXi gammaSource(sourceCount);
    Eigen::VectorXi gammaSink(sinkCount);
    sourceCount = 0;
    sinkCount = 0;
    for (auto node : tunnelCycles.back()) {
        gammaSource(sourceCount++) = node;
    }
    for (auto i = 0; i < tunnelCycles.size()-1; i++) {
        for (auto node : tunnelCycles[i]) {
            gammaSink(sinkCount++) = node;
        }
    }

    igl::heat_geodesics_solve(data, gammaSource, DSource);
    igl::heat_geodesics_solve(data, gammaSink, DSink);

    
    for (auto i = 0; i < EV.rows(); i++) {
        if (DSource(EV(i, 0)) > DSink(EV(i, 0)) && DSink(EV(i, 1)) > DSource(EV(i, 1))) {
            cutEdgeSet.insert(i);
            viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            //Sviewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), green);
        }
        else if (DSource(EV(i, 0)) < DSink(EV(i, 0)) && DSink(EV(i, 1)) < DSource(EV(i, 1))) {
            cutEdgeSet.insert(i);
            viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            //viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), green);
        }
    }
}

void launch_viewer() {
    viewer.data().set_mesh(V, F);
    viewer.data().show_lines = true;
    viewer.launch();
}


int main(int argc, char* argv[])
{
    std::string filename = "../resources/3holes.off";
    
    if (FERTILITY) {
        filename = "../resources/fertility.off";
    }
    if (argc > 1)
    {
        filename = argv[1];
    }
    // Load a mesh in OFF format
    igl::read_triangle_mesh(filename, V, F);
    igl::heat_geodesics_precompute(V, F, data);
    // Compute curvature directions via quadric fitting
    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
    igl::edge_topology(V, F, EV, FE, EF);
    igl::barycenter(V, F, FCV); // Face Center Vertex

    DEW.resize(EF.rows(), 2);//Dual Edge weights
    EVW.resize(EV.rows(), 2);//Primary Edge weights
    for (auto i = 0; i < V.rows(); i++) {
        std::unordered_map<int, int> uom;
        VVE.push_back(uom);
    }
    for (auto i = 0; i < FCV.rows(); i++) {
        std::unordered_map<int, int> uom;
        dVVE.push_back(uom);
    }

    int g = (EV.rows() - V.rows() - F.rows() + 2) / 2; //genus




    //viewer.data().set_mesh(V, F);
    std::vector<WeightedEdge> primalEdgesMin;
    std::vector<WeightedEdge> primalEdgesMax;
    std::vector<WeightedEdge> dualEdgesMin;
    std::vector<WeightedEdge> dualEdgesMax;

    std::unordered_set<int> uos;
    for (auto i = 0; i < EV.rows(); i++) {
        Eigen::VectorXd edgeD = V.row(EV(i, 1)) - V.row(EV(i, 0));
        edgeD.normalize();
        EVW.row(i) << abs(edgeD.dot(PD1.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD1.row(EV(i, 1)))) / 2,
            abs(edgeD.dot(PD2.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD2.row(EV(i, 1)))) / 2;
        primalEdgesMin.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 0)));
        primalEdgesMax.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 1)));
        VVE[EV(i, 0)][EV(i, 1)] = i;
        VVE[EV(i, 1)][EV(i, 0)] = i;
    }


    for (auto i = 0; i < EF.rows(); i++) {
        dualEdgesMin.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), 1 - EVW(i, 0)));
        dualEdgesMax.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), 1 - EVW(i, 1)));
        dVVE[EF(i, 0)][EF(i, 1)] = i;
        dVVE[EF(i, 1)][EF(i, 0)] = i;
    }

    int ngc = 0;
    int MAX = 5 * g;
    int iter = 0;
    bool flag = 0;
    while (goodCycles.size() < 2 * g && iter < MAX) {
        std::vector<std::vector<int>> cycles;
        if (iter % 2 == 1) {
            tree_cotree(primalEdgesMin, dualEdgesMin, cycles, viewer);

            find_good_cycles(cycles, viewer);
            if (DEBUG) {
                visualize_cycle(viewer, goodCycles.back(), blue);
                break;
            }
        }
        else {
            tree_cotree(primalEdgesMax, dualEdgesMax, cycles, viewer);
            if (primalSpanViewFlag || dualSpanViewFlag) {
                launch_viewer();
                return 0;
            }
            find_good_cycles(cycles, viewer);
            if (DEBUG) {
                visualize_cycles(viewer, cycles, blue);
                break;
            }
        }
        ++iter;
        if (DEBUG)
            break;
    }

    
    get_tunnel_cycles(goodCycles, tunnelCycles);

    visualize_cycles(viewer, tunnelCycles, blue);
    if(VISUALIZEHANDLES)
        visualize_cycles(viewer, goodCycles, red);

    if (CUTMESH) {
        std::unordered_set<int> cutEdgeSet;
        reverse(tunnelCycles.begin(), tunnelCycles.end());
        while (tunnelCycles.size() != 1) {
            get_cut_edges(tunnelCycles, cutEdgeSet);
            tunnelCycles.pop_back();
            cut_mesh(V, F, cutEdgeSet);
            break;
        }
    }

    // Draw a blue segment parallel to the minimal curvature direction
    // const double avg = igl::avg_edge_length(V, F);
    //viewer.data().add_edges(V + PD2 * avg, V - PD2 * avg, blue);
    //viewer.data().add_edges(V + PD1 * avg, V - PD1 * avg, red);

    launch_viewer();
    return 0;
}

float DistToCentroid(std::vector<int>& cycle)
{

    Eigen::MatrixXd cycleV(cycle.size(), 3);
    Eigen::MatrixXi cycleF(1, cycle.size());
    Eigen::MatrixXd BC;

    for (unsigned int i = 0; i < cycle.size(); ++i)
    {
        cycleV.row(i) = V.row(cycle[i]);
        cycleF(0, i) = i;
    }
    igl::barycenter(cycleV, cycleF, BC);

    float totalDistance = 0.0f;

    for (unsigned int i = 0; i < cycleV.rows(); ++i)
    {
        totalDistance += (cycleV.row(i) - BC.row(0)).norm();
    }

    return 0.04 * totalDistance;
}

bool CompareCycles(std::vector<int>& a, std::vector<int>& b)
{
    float pathCostA = 0.0f;
    float pathCostB = 0.0f;

    pathCostA += a.size();
    pathCostB += b.size();

    pathCostA += DistToCentroid(a);
    pathCostB += DistToCentroid(b);

    return pathCostA < pathCostB;

}