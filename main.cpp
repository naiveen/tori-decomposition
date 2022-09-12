
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
#include <igl/barycenter.h>
#include <igl/adjacency_matrix.h>
#include <fstream>
#include <vector>
#include <forward_list>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"
#include <stack>

bool DEBUG = 0;
igl::opengl::glfw::Viewer viewer;
/*
Eigen::MatrixXi TT;
Eigen::MatrixXi TTi;
Eigen::MatrixXi E;
Eigen::MatrixXi uE;
Eigen::MatrixXi EMAP;
std::vector<std::vector<int>> uE2E;
*/
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
void tree_cotree(std::vector<WeightedEdge>& primalEdges, std::vector<WeightedEdge>& dualEdges, std::vector<std::vector<int>>& cycles, igl::opengl::glfw::Viewer& viewer) {
    /*
    * Given Primary and Dual edges, the function computes 2*g good cycles.
    */


    std::vector<WeightedEdge>  primalSpan;
    std::vector<WeightedEdge> dualSpan;
    std::vector<std::vector<int>>VE(V.rows(), std::vector<int>());
    std::unordered_set<int> uos;

    //Assign low weight to edges that form good cycles
    for (auto& index : goodEdgeIndexVec) {
        primalEdges[index].dist = 0;
        dualEdges[index].dist = 0;
    }
    for (auto& index : badEdgeIndexVec) {
        primalEdges[index].dist = INT_MAX;
        dualEdges[index].dist = INT_MAX;
    }
    kruskalMST(V.rows(), primalEdges, primalSpan);

    bool flag = 0;
    //Add vertex to edge relation for edges in spanning tree
    for (auto edge : primalSpan) {
        uos.insert(edge.index);
        VE[edge.dst].push_back(edge.src);
        VE[edge.src].push_back(edge.dst);
        if (flag)
            viewer.data().add_edges(V.row(edge.src), V.row(edge.dst), green);
    }

    //Assign high weight to dual graph edges corresponding to primary graph edges
    for (auto& index : uos) {
        dualEdges[index].dist = INT_MAX;
    }

    //Spanning tree computation on dual Graph
    kruskalMST(FCV.rows(), dualEdges, dualSpan);
    //Reset dual graph edge weights back
    for (auto& index : uos) {
        dualEdges[index].dist = dualEdges[index]._dist;
    }

    for (auto& index : goodEdgeIndexVec) {
        primalEdges[index].dist = primalEdges[index]._dist;
        dualEdges[index].dist = dualEdges[index]._dist;
    }
    //Add dual graph edges to indexes to set
    for (auto edge : dualSpan) {
        uos.insert(edge.index);
        if (flag) {
            //viewer.data().add_edges(V.row(EV(edge.index, 0)), V.row(EV(edge.index, 1)), red);
            //viewer.data().add_edges(FCV.row(edge.src), (V.row(EV(edge.index, 0)) + V.row(EV(edge.index, 1))) / 2, red);
            //viewer.data().add_edges(FCV.row(edge.dst), (V.row(EV(edge.index, 0)) + V.row(EV(edge.index, 1))) / 2, red);
        }
    }

    //Cycle detection

    for (auto i = 0; i < EV.rows(); i++) {
        if (uos.count(i) == 0) {

            std::vector<int> cycle;
            if (flag)
                viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), red);
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

void visualize_cycle(igl::opengl::glfw::Viewer& viewer, std::vector<int>& cycle, Eigen::RowVector3d color) {

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


/*
void one_ring_tight(std::vector<int>& cycle, std::vector<int>& newCycle, igl::opengl::glfw::Viewer& viewer) {
    std::unordered_set<int> start_strip;
    get_start_strip(cycle, start_strip);
    std::unordered_set<int> prev_nodes;
    get_start_strip(cycle, prev_nodes);
    for (auto it = VVE[cycle[0]].begin(); it != VVE[cycle[0]].end(); it++) {
        if (start_strip.count(it->first) != 0) {
            viewer.data().add_edges(V.row(cycle[0]), V.row(it->first), blue);
        }
    }
    int k = 2;
    std::unordered_set<int> curr_nodes;
    for (auto i = 1; i < cycle.size(); i++) {
        curr_nodes.clear();
        curr_nodes.insert(cycle[i]);
        for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
            for (auto it2 = VVE[it1->first].begin(); it2 != VVE[it1->first].end(); it2++) {
                if (prev_nodes.count(it2->first) != 0) {
                    curr_nodes.insert(it1->first);
                    if (i == k) {
                        viewer.data().add_edges(V.row(it1->first), V.row(it2->first), green);
                    }
                    else if (i == k - 1) {
                        viewer.data().add_edges(V.row(it1->first), V.row(it2->first), red);
                    }
                    else {
                        viewer.data().add_edges(V.row(it1->first), V.row(it2->first), blue);
                    }
                }
            }

            if (i == k ) {
                viewer.data().add_edges(V.row(it1->first), V.row(cycle[i]), green);
            }
            else if (i == k - 1) {
                viewer.data().add_edges(V.row(it1->first), V.row(cycle[i]), red);
            }
            else {
                viewer.data().add_edges(V.row(it1->first), V.row(cycle[i]), blue);
            }
        }
        prev_nodes.clear();
        for (auto it = curr_nodes.begin(); it != curr_nodes.end(); it++) {
            prev_nodes.insert(*it);
        }
        if (i == k)
            return;

    }

    return;

    float best_cycle_metric = INT_MAX;
    float temp_metric;
    for (auto root_node : start_strip) {
        std::vector<int> tempCycle;
        //temp_metric = dijkstra_cycle(root_node, cycle, tempCycle);
        if (best_cycle_metric > temp_metric) {
            best_cycle_metric = temp_metric;
            newCycle = tempCycle;
        }
    }
}
*/

/*
void one_ring_tight(std::vector<int>& cycle, std::vector<int>& newCycle, igl::opengl::glfw::Viewer& viewer) {
    std::unordered_set<int> start_strip;
    std::unordered_set<int> end_strip;
    std::unordered_set<int> ring;







    start_strip.erase(cycle.back());
    for (auto it = VVE[cycle[1]].begin(); it != VVE[cycle[1]].end(); it++) {
        ring.insert(it->first);
    }
    ring.insert(cycle[1]);
    ring.erase(cycle[0]);
    std::vector<int> temp;
    for (auto it1 = start_strip.begin(); it1 != start_strip.end(); it1++) {
        int currVertex = *it1;
        int flag = 0;
        for (auto it2 = ring.begin(); it2 != ring.end(); it2++) {
            if (VVE[currVertex].count(*it2) != 0) {
                flag = 1;
                break;
            }
        }
        if (!flag) {
            temp.push_back(currVertex);
        }
    }
    for (auto each : temp) {
        start_strip.erase(each);
    }


    for (auto it = VVE[cycle.back()].begin(); it != VVE[cycle.back()].end(); it++) {
        end_strip.insert(it->first);
    }
    end_strip.insert(cycle.back());
    end_strip.erase(cycle[0]);
    end_strip.erase(cycle[cycle.size() - 2]);
    ring.clear();
    for (auto it = VVE[cycle[0]].begin(); it != VVE[cycle[0]].end(); it++) {
        ring.insert(it->first);
    }
    ring.insert(cycle[0]);
    ring.erase(cycle.back());
    std::vector<int> temp2;
    for (auto it1 = end_strip.begin(); it1 != end_strip.end(); it1++) {
        int currVertex = *it1;
        int flag = 0;
        for (auto it2 = ring.begin(); it2 != ring.end(); it2++) {
            if (VVE[currVertex].count(*it2) != 0) {
                flag = 1;
                break;
            }
        }
        if (!flag) {
            temp2.push_back(currVertex);
        }
    }
    for (auto each : temp2) {
        end_strip.erase(each);
    }
    for (auto it = end_strip.begin(); it != end_strip.end(); it++) {
        viewer.data().add_edges(V.row(cycle.back()), V.row(*it), red);
    }



    std::unordered_map<int, std::pair<int, float>> m;

    for (auto it = start_strip.begin(); it != start_strip.end(); it++) {
        m[*it].first = *it;
        m[*it].second = 0;
    }
    int k = 1;
    for (auto i = 1; i < cycle.size(); i++) {
        //viewer.data().add_edges(V.row(cycle[i-1]), V.row(cycle[i]), green);
        if (m.count(cycle[i]) == 0) {
            m[cycle[i]].first = cycle[i - 1];
            m[cycle[i]].second = m[cycle[i - 1]].second + get_tight_cycle_metric(cycle[i - 1], cycle[i]);
        }

        for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
            int currVertex = it1->first;
            if (m.count(currVertex) == 0) {
                m[currVertex].first = cycle[i];
                m[currVertex].second = m[cycle[i]].second + get_tight_cycle_metric(cycle[i], currVertex);
            }
            for (auto it2 = VVE[currVertex].begin(); it2 != VVE[currVertex].end(); it2++) {
                if (i != 1 && start_strip.count(it2->first) != 0) {
                    continue;
                }
                if (it2->first != cycle[i] && m.count(it2->first) != 0 &&
                    get_tight_cycle_metric(currVertex, it2->first) + m[it2->first].second < m[currVertex].second) {
                    m[currVertex].second = get_tight_cycle_metric(currVertex, it2->first) + m[it2->first].second;
                    m[currVertex].first = it2->first;
                    if (m[cycle[i]].second > get_tight_cycle_metric(cycle[i], currVertex) + m[currVertex].second) {
                        m[cycle[i]].first = currVertex;
                        m[cycle[i]].second = get_tight_cycle_metric(cycle[i], currVertex) + m[currVertex].second;
                    }
                }
            }
        }
        for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
            int currVertex = it1->first;
            if (get_tight_cycle_metric(currVertex, cycle[i]) + m[cycle[i]].second < m[currVertex].second) {
                m[currVertex].second = get_tight_cycle_metric(currVertex, cycle[i]) + m[cycle[i]].second;
                m[currVertex].first = cycle[i];
            }

            if (m[currVertex].second == 0) {
                std::cout << currVertex << " "<<m[currVertex].first << std::endl;
                viewer.data().add_edges(V.row(currVertex), V.row(m[currVertex].first), red);
            }
            if (i == k) {
                //viewer.data().add_edges(V.row(currVertex), V.row(m[currVertex].first), red);
            }
            else {
                //viewer.data().add_edges(V.row(currVertex), V.row(m[currVertex].first), blue);
            }
        }
    }
    /*
    int i = cycle.size() - 1;
    if (m.count(cycle[i]) == 0) {
        m[cycle[i]].first = cycle[i - 1];
        m[cycle[i]].second = m[cycle[i - 1]].second + get_tight_cycle_metric(cycle[i - 1], cycle[i]);
    }
    for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
        int currVertex = it1->first;
        if (start_strip.count(currVertex) != 0)
            continue;
        if (m.count(it1->first) == 0) {
            m[currVertex].first = cycle[i];
            m[currVertex].second = get_tight_cycle_metric(cycle[i], currVertex) + m[cycle[i]].second;
        }
        for (auto it2 = VVE[currVertex].begin(); it2 != VVE[currVertex].end(); it2++) {
            if (it2->first != cycle[i] && m.count(it2->first) != 0 &&
                get_tight_cycle_metric(currVertex, it2->first) + m[it2->first].second < m[currVertex].second) {
                m[currVertex].second = get_tight_cycle_metric(currVertex, it2->first) + m[it2->first].second;
                m[currVertex].first = it2->first;
                if (m[cycle[i]].second > get_tight_cycle_metric(cycle[i], currVertex) + m[currVertex].second) {
                    m[cycle[i]].first = currVertex;
                    m[cycle[i]].second = get_tight_cycle_metric(cycle[i], currVertex) + m[currVertex].second;
                }
            }
        }
    }
    for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
        int currVertex = it1->first;
        if (start_strip.count(currVertex) != 0)
            continue;
        if (get_tight_cycle_metric(currVertex, cycle[i]) + m[cycle[i]].second < m[currVertex].second) {
            m[currVertex].second = get_tight_cycle_metric(currVertex, cycle[i]) + m[cycle[i]].second;
            m[currVertex].first = cycle[i];
        }
    }

    int bestCyclePos = cycle.back();
    float best_cycle_metric = m[cycle.back()].second;
    for (auto it = VVE[cycle.back()].begin(); it != VVE[cycle.back()].end(); it++) {
        if (best_cycle_metric < m[it->first].second) {
            bestCyclePos = it->first;
            best_cycle_metric = m[it->first].second;
        }
    }

    int nextPos = m[bestCyclePos].first;
    newCycle.push_back(bestCyclePos);
    while (m[nextPos].second != 0) {
        newCycle.push_back(nextPos);
        nextPos = m[nextPos].first;
    }
    newCycle.push_back(nextPos);



    if (VVE[nextPos].count(bestCyclePos) == 0) {
        float mtemp = INT_MAX;
        int finalPos = 0;
        for (auto it = start_strip.begin(); it != start_strip.end(); it++) {
            if (VVE[nextPos].count(*it) && VVE[bestCyclePos].count(*it)) {
                float temp = get_tight_cycle_metric(nextPos, *it) + get_tight_cycle_metric(bestCyclePos, *it);
                if (temp < mtemp) {
                    mtemp = temp;
                    finalPos = *it;
                }
            }
        }
        newCycle.push_back(finalPos);
    }


}
*/

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
            //it2 = it1;
            //it2++;
            //it3 = it2;
            //it3++;
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
void one_ring_tight_scheduler(std::vector<int>& cycle, std::vector<int>& newCycle, igl::opengl::glfw::Viewer& viewer) {
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

    /*
    while (1) {
        if (DEBUG) {
            DEBUG = 0;
            break;
        }
        if (cycle1.size() == 500)
            break;
        for (auto it = VVE[cycle1.back()].begin(); it != VVE[cycle1.back()].end(); it++) {
            if (main_cycle_set.count(it->first)==0&& node_set.count(it->first) != 0 && it->first != cycle1[cycle1.size() - 2]) {
                cycle1.push_back(it->first);
                break;
            }
        }
        if (VVE[cycle1[0]].count(cycle1.back()) != 0)
            break;
    }

    while (1) {
        if (cycle2.size() == 500)
            break;
        for (auto it = VVE[cycle2.back()].begin(); it != VVE[cycle2.back()].end(); it++) {
            if (main_cycle_set.count(it->first) == 0 && node_set.count(it->first) != 0 && it->first != cycle2[cycle2.size() - 2]) {
                cycle2.push_back(it->first);
                break;
            }
        }
        if (VVE[cycle2[0]].count(cycle2.back()) != 0)
            break;
    }
    */

    /*
    for (auto it = start_strip.begin(); it != start_strip.end(); it++) {
        dist_map[*it].first = cycle[0];
        dist_map[*it].second = get_tight_cycle_metric(cycle[0], *it);
        if (DEBUG)
            viewer.data().add_edges(V.row(*it), V.row(cycle[0]), blue);
    }
    DEBUG = 0;
    if (DEBUG) {
        int k = 4;
        for (auto i = 1; i < cycle.size(); i++) {
            if (i == k)
                break;
            viewer.data().add_edges(V.row(cycle[i - 1]), V.row(cycle[i]), blue);
        }
        for (auto i = 1; i < cycle.size(); i++) {
            if (i == k)
                break;
            for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
                viewer.data().add_edges(V.row(it1->first), V.row(cycle[i]), green);
            }
        }
        for (auto i = 1; i < cycle.size(); i++) {
            if (i == k)
                break;
            for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
                for (auto it2 = VVE[it1->first].begin(); it2 != VVE[it1->first].end(); it2++) {
                    if (cycle_edges.count(it2->second) == 0)
                        continue;
                    if (i != 1 && start_strip.count(it2->first) != 0) {
                        continue;
                    }
                    if (it2->first != cycle[i] && dist_map.count(it2->first) != 0) {
                        viewer.data().add_edges(V.row(it1->first), V.row(cycle[i]), red);
                    }
                }
            }
        }
    }

    for (auto i = 1; i < cycle.size(); i++) {
        //viewer.data().add_edges(V.row(cycle[i-1]), V.row(cycle[i]), green);
        if (dist_map.count(cycle[i]) == 0) {
            dist_map[cycle[i]].first = cycle[i - 1];
            dist_map[cycle[i]].second = dist_map[cycle[i - 1]].second + get_tight_cycle_metric(cycle[i - 1], cycle[i]);
        }

        for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
            int currVertex = it1->first;
            if (dist_map.count(currVertex) == 0) {
                dist_map[currVertex].first = cycle[i];
                dist_map[currVertex].second = dist_map[cycle[i]].second + get_tight_cycle_metric(cycle[i], currVertex);
            }
            for (auto it2 = VVE[currVertex].begin(); it2 != VVE[currVertex].end(); it2++) {
                if (cycle_edges.count(it2->second) == 0)
                    continue;
                if (i != 1 && start_strip.count(it2->first) != 0) {
                    continue;
                }
                if (it2->first != cycle[i] && dist_map.count(it2->first) != 0 &&
                    get_tight_cycle_metric(currVertex, it2->first) + dist_map[it2->first].second < dist_map[currVertex].second) {
                    dist_map[currVertex].second = get_tight_cycle_metric(currVertex, it2->first) + dist_map[it2->first].second;
                    dist_map[currVertex].first = it2->first;
                    if (dist_map[cycle[i]].second > get_tight_cycle_metric(cycle[i], currVertex) + dist_map[currVertex].second) {
                        dist_map[cycle[i]].first = currVertex;
                        dist_map[cycle[i]].second = get_tight_cycle_metric(cycle[i], currVertex) + dist_map[currVertex].second;
                    }
                }
            }
        }
        for (auto it1 = VVE[cycle[i]].begin(); it1 != VVE[cycle[i]].end(); it1++) {
            int currVertex = it1->first;
            if (get_tight_cycle_metric(currVertex, cycle[i]) + dist_map[cycle[i]].second < dist_map[currVertex].second) {
                dist_map[currVertex].second = get_tight_cycle_metric(currVertex, cycle[i]) + dist_map[cycle[i]].second;
                dist_map[currVertex].first = cycle[i];
            }
        }
    }

    int bestCyclePos = cycle[0];
    float bestMetric = INT_MAX;
    for (auto it = VVE[cycle[0]].begin(); it != VVE[cycle[0]].end(); it++) {
        if (start_strip.count(it->first) != 0 || it->first == cycle[1] || dist_map.count(it->first) == 0)
            continue;
        if (bestMetric > dist_map[it->first].second + get_tight_cycle_metric(it->first, cycle[0])) {
            bestMetric = dist_map[it->first].second + get_tight_cycle_metric(it->first, cycle[0]);
            bestCyclePos = it->first;
        }
    }
    if (bestCyclePos == cycle[0]) {
        newCycle = cycle;
        return;
    }
    int nextPos = bestCyclePos;
    while (dist_map[nextPos].first != cycle[0]) {
        newCycle.push_back(nextPos);
        nextPos = dist_map[nextPos].first;
    }
    newCycle.push_back(nextPos);
    newCycle.push_back(cycle[0]);
    reverse(newCycle.begin(), newCycle.end());
    if (DEBUG) {
        visualize_cycle(viewer, cycle, blue);
        visualize_cycle(viewer, newCycle, red);
        float cycle1_metric = 0;
        float cycle2_metric = 0;
        for (auto i = 1; i < cycle.size(); i++) {
            cycle1_metric += get_tight_cycle_metric(cycle[i - 1], cycle[i]);

        }
        cycle1_metric += get_tight_cycle_metric(cycle[0], cycle.back());
        for (auto i = 1; i < newCycle.size(); i++) {
            cycle2_metric += get_tight_cycle_metric(newCycle[i - 1], newCycle[i]);

        }
        cycle2_metric += get_tight_cycle_metric(newCycle[0], newCycle.back());
        return;
    }
    */
}

void tighten_cycle(std::vector<int>& cycle, std::vector<int>& newCycle, igl::opengl::glfw::Viewer& viewer) {
    int count = 0;
    edge_tight_cycle(cycle);
    //one_ring_tight_scheduler(cycle, newCycle, viewer);
    /*
    while (count<5 && get_cycle_diff(cycle, newCycle) != 0) {
        DEBUG = 0;
        count++;
        one_ring_tight_scheduler(cycle, newCycle, viewer);

        edge_tight_cycle(newCycle);

        if (DEBUG) {
            visualize_cycle(viewer, newCycle, red);
        }
        if (DEBUG) {
            visualize_cycle(viewer, cycle, blue);
        }
        if (DEBUG)
            return;
        cycle = newCycle;
        newCycle.clear();
    }
    */
}

/*
*
* Sample version based on length of cycles; */
void find_good_cycles(std::vector<std::vector<int>>& cycles, igl::opengl::glfw::Viewer& viewer) {
    //Find a good cycle and store it in goodCycles
    std::vector<std::vector<int>> uniqueCycles;
    for (auto cycle : cycles) {
        edge_tight_cycle(cycle);
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
        std::vector<int> newCycle;
        int count = 0;
        while (count < 50) {
            newCycle.clear();
            one_ring_tight_scheduler(uniqueCycles[0], newCycle, viewer);
            if (DEBUG) {
                return;
            }
            if (get_cycle_diff(uniqueCycles[0], newCycle) == 0) {
                break;
            }
            edge_tight_cycle(newCycle);
            uniqueCycles[0] = newCycle;
            count++;

        }
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

}

void cut_graph(std::vector<std::vector<int>>& goodCycles, igl::opengl::glfw::Viewer& viewer) {

    /*
    Min cut on graph. Need to update source weights and sink weights

    */
    typedef Graph<int, int, int> GraphType;
    GraphType* g = new GraphType(V.rows(), EV.rows());

    for (auto i = 0; i < V.rows(); i++) {
        g->add_node();
    }
    for (auto i = 0; i < EV.rows(); i++) {
        g->add_edge(EV(i, 0), EV(i, 1), 10, 10);
    }
    for (auto j = 0; j < goodCycles[0].size(); j++) {
        g->add_tweights(goodCycles[0][j], 1000, -1000);
    }
    for (auto i = 1; i < goodCycles.size(); i++) {
        for (auto j = 0; j < goodCycles[i].size(); j++) {
            g->add_tweights(goodCycles[i][j], -1000, 1000);
        }
    }
    int flow = g->maxflow();
    for (auto i = 0; i < EV.rows(); i++) {
        if (g->what_segment(EV(i, 0)) == GraphType::SOURCE && g->what_segment(EV(i, 1)) == GraphType::SINK) {
            viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            //viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), green);
        }
        else if (g->what_segment(EV(i, 0)) == GraphType::SINK && g->what_segment(EV(i, 1)) == GraphType::SOURCE) {
            viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
            //viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), green);
        }
    }
    delete g;
}

int main(int argc, char* argv[])
{
    std::string filename = "../resources/3holes.off";
    int fertility = 1;
    if (fertility) {
        filename = "../resources/fertility.off";
    }
    if (argc > 1)
    {
        filename = argv[1];
    }
    // Load a mesh in OFF format
    igl::read_triangle_mesh(filename, V, F);
    igl::edge_topology(V, F, EV, FE, EF);
    igl::barycenter(V, F, FCV); // Face Center Vertex

    DEW.resize(EF.rows(), 2);//Dual Edge weights
    EVW.resize(EV.rows(), 2);//Primary Edge weights
    for (auto i = 0; i < V.rows(); i++) {
        std::unordered_map<int, int> uom;
        VVE.push_back(uom);
    }

    int g = (EV.rows() - V.rows() - F.rows() + 2) / 2; //genus


    // Compute curvature directions via quadric fitting
    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd PV1, PV2;
    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);



    viewer.data().set_mesh(V, F);
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
    }

    int ngc = 0;
    int MAX = 5 * g;
    int iter = 0;
    bool flag = 0;
    while (goodCycles.size() < 2 * g && iter < MAX) {
        std::vector<std::vector<int>> cycles;
        DEBUG = 0;
        if (iter % 2 == 0) {
            tree_cotree(primalEdgesMin, dualEdgesMin, cycles, viewer);
            if (DEBUG) {
                visualize_cycles(viewer, cycles, blue);
                break;
            }
            find_good_cycles(cycles, viewer);
            visualize_cycle(viewer, goodCycles.back(), blue);
        }
        else {
            tree_cotree(primalEdgesMax, dualEdgesMax, cycles, viewer);
            if (DEBUG) {
                visualize_cycles(viewer, cycles, blue);
                break;
            }
            find_good_cycles(cycles, viewer);
            visualize_cycle(viewer, goodCycles.back(), red);
            //visualize_cycles(viewer, cycles, red);
        }
        ++iter;
        if (DEBUG)
            break;
    }
    if (!DEBUG)
        visualize_cycles(viewer, goodCycles, blue);
    //get_tunnel_cycles();
    //cut_graph(goodCycles, viewer);


    // Draw a blue segment parallel to the minimal curvature direction
    // const double avg = igl::avg_edge_length(V, F);
    //viewer.data().add_edges(V + PD2 * avg, V - PD2 * avg, blue);
    //viewer.data().add_edges(V + PD1 * avg, V - PD1 * avg, red);


    viewer.data().show_lines = false;
    viewer.launch();
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