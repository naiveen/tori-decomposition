
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
//#include <igl/edges.h>
//#include <igl/triangle_triangle_adjacency.h>
//#include <igl/flip_edge.h>
//#include <igl/unique_edge_map.h>
#include <igl/edge_topology.h>
#include <igl/barycenter.h>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"
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
Eigen::MatrixXi EV;
Eigen::MatrixXi FE;
Eigen::MatrixXi EF; //Also dual graph edge vertices in FCV
Eigen::MatrixXd FCV;
Eigen::MatrixXd DEW;
Eigen::MatrixXd EVW;

std::vector<std::vector<int>> goodCycles;
std::unordered_set<int> goodEdgeIndexVec;
std::vector<std::unordered_map<int,int>> VVE;

const Eigen::RowVector3d red(0.8, 0.2, 0.2), blue(0.2, 0.2, 0.8), green(0.2,0.8,0.2);

struct WeightedEdge {
    float dist = 0,_dist=0;
    int src, dst,index;
    WeightedEdge( int index, int src, int dst, float dist) : index(index), src(src), dst(dst), dist(dist),_dist(dist) {};
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

void dfs(std::vector<std::vector<int>>& VE, std::vector<int>&cycle, int prevIndex, int curIndex, bool &cycleflag) {

    if (curIndex == cycle[0]) {
        cycleflag = true;
        return;
    }
    for (auto i = 0; i < VE[curIndex].size(); i++) {
        if (VE[curIndex][i] == prevIndex)
            continue;
        dfs(VE, cycle, curIndex, VE[curIndex][i], cycleflag );
        if (cycleflag) {
            cycle.push_back(curIndex);
            break;
        }
    }
    return;
}

void tree_cotree(std::vector<WeightedEdge> & primalEdges, std::vector<WeightedEdge>& dualEdges, std::vector<std::vector<int>>& cycles, igl::opengl::glfw::Viewer& viewer) {
    /*
    * Given Primary and Dual edges, the function computes 2*g good cycles.
    */


    std::vector<WeightedEdge>  primalSpan;
    std::vector<WeightedEdge> dualSpan;
    std::vector<std::vector<int>>VE(V.rows(), std::vector<int>());
    std::unordered_set<int> uos;

    //Assign low weight to edges that form good cycles
    for (auto &index : goodEdgeIndexVec) {
        primalEdges[index].dist = 0;
        dualEdges[index].dist = 0;
    }
    kruskalMST(V.rows(), primalEdges, primalSpan);
    
    bool flag = 0;
    //Add vertex to edge relation for edges in spanning tree
    for (auto edge : primalSpan) {
        uos.insert(edge.index);
        VE[edge.dst].push_back(edge.src);
        VE[edge.src].push_back(edge.dst);
        if(flag)
            viewer.data().add_edges(V.row(edge.src), V.row(edge.dst), blue);
    }

    //Assign high weight to dual graph edges corresponding to primary graph edges
    for (auto& index :uos)  {
        dualEdges[index].dist = INT_MAX;
    }

    //Spanning tree computation on dual Graph
    kruskalMST(FCV.rows(), dualEdges, dualSpan);
    //Reset dual graph edge weights back
    for (auto& index : uos) {
        dualEdges[index].dist = dualEdges[index]._dist;
    }

    for (auto &index : goodEdgeIndexVec) {
        primalEdges[index].dist = primalEdges[index]._dist;
        dualEdges[index].dist = dualEdges[index]._dist;
    }
    //Add dual graph edges to indexes to set
    for (auto edge : dualSpan) {
        uos.insert(edge.index);
        if (flag) {
            viewer.data().add_edges(FCV.row(edge.src), (V.row(EV(edge.index,0)) + V.row(EV(edge.index, 1))) / 2, red);
            viewer.data().add_edges(FCV.row(edge.dst), (V.row(EV(edge.index, 0)) + V.row(EV(edge.index, 1))) / 2, red);
        }
    }

    //Cycle detection

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
void visualize_cycles(igl::opengl::glfw::Viewer &viewer, std::vector<std::vector<int>> &cycles, Eigen::RowVector3d color) {
    int i = 0;
    sort(cycles.begin(), cycles.end(), [](std::vector<int>& a, std::vector<int>& b) -> bool {
        return a.size() < b.size();
        });
    for (auto cycle : cycles) {
        if (i == 3)
            break;
        i++;
        viewer.data().add_edges(V.row(cycle[0]), V.row(cycle.back()), color);
        for (auto j = 1; j < cycle.size(); j++) {
            viewer.data().add_edges(V.row(cycle[j - 1]), V.row(cycle[j]), color);
        }
    }
}


void find_good_cycles(std::vector<std::vector<int>>& cycles) {
}
/*
* 
* Sample version based on length of cycles;
void find_good_cycles(std::vector<std::vector<int>> &cycles) {
    //Find a good cycle and store it in goodCycles
    std::vector<std::vector<int>> uniqueCycles;
    for (auto cycle: cycles) {
        bool flag = 0;
        if (goodEdgeIndexVec.count(VVE[cycle[0]][cycle.back()]) > 0) {
            flag = 1;
        }
        for (auto j = 1; j < cycle.size(); j++) {
            if (goodEdgeIndexVec.count(VVE[cycle[j - 1]][cycle[j]]) > 0) {
                flag = 1;
                break;
            }
        }
        if (!flag) {
            uniqueCycles.push_back(cycle);
            flag = 0;
        }
    }
    sort(uniqueCycles.begin(), uniqueCycles.end(), [](std::vector<int>& a, std::vector<int>& b) -> bool {
        return a.size() < b.size();
        });
    if (uniqueCycles.size() > 0) {
        goodCycles.push_back(uniqueCycles[0]);
        int goodCycleIndex = 0;
        //Add edge index of the good cycle to goodEdgeIndexVec
        for (auto j = 1; j < uniqueCycles[0].size(); j++) {
            goodEdgeIndexVec.insert(VVE[uniqueCycles[0][j - 1]][uniqueCycles[0][j]]);
        }
        goodEdgeIndexVec.insert(VVE[uniqueCycles[0][0]][uniqueCycles[0].back()]);
    }
}
*/
void cut_graph(std::vector<std::vector<int>>& goodCycles, igl::opengl::glfw::Viewer& viewer ) {

    /*
    Min cut on graph. Need to update source weights and sink weights
    
    */
    typedef Graph<int, int, int> GraphType;
    GraphType* g = new GraphType( V.rows(), EV.rows());

    for (auto i=0; i < V.rows(); i++) {
        g->add_node();
    }
    for (auto i = 0; i < EV.rows(); i++) {
        g->add_edge(EV(i,0),EV(i,1), 10,10);
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
        if (g->what_segment(EV(i,0)) == GraphType::SOURCE && g->what_segment(EV(i, 1)) == GraphType::SINK) {
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


/*
void cut_graph(std::vector<std::vector<int>>& goodCycles, igl::opengl::glfw::Viewer& viewer) {
    typedef Graph<int, int, int> GraphType;
    GraphType* g = new GraphType(FCV.rows(), EV.rows());

    for (auto i = 0; i < FCV.rows(); i++) {
        g->add_node();
    }
    for (auto i = 0; i < EF.rows(); i++) {
        g->add_edge(EF(i, 0), EF(i, 1), 10, 10);
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
            viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), red);
        }
        else if (g->what_segment(EV(i, 0)) == GraphType::SINK && g->what_segment(EV(i, 1)) == GraphType::SOURCE) {
            viewer.data().add_edges(V.row(EV(i, 0)), V.row(EV(i, 1)), red);
        }
    }

    delete g;

}
*/
int main(int argc, char *argv[])
{
  std::string filename = "../resources/3holes.off";
  if(argc>1)
  {
    filename = argv[1];
  }
  // Load a mesh in OFF format
  igl::read_triangle_mesh(filename, V, F);
  igl::edge_topology(V, F,EV,FE,EF );
  igl::barycenter(V, F, FCV); // Face Center Vertex

  DEW.resize(EF.rows(), 2);//Face Center Vertex1 Index, Face Center Vertex2 Index
  EVW.resize(EV.rows(), 2);
  for (auto i = 0; i < V.rows(); i++) {
      std::unordered_map<int, int> uom;
      VVE.push_back(uom);
  }
  
  int g = (EV.rows() - V.rows() - F.rows() + 2)/2;
  // Compute curvature directions via quadric fitting
  Eigen::MatrixXd PD1, PD2;
  Eigen::VectorXd PV1, PV2;
  igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  const Eigen::RowVector3d red(0.8, 0.2, 0.2), blue(0.2, 0.2, 0.8);
  std::vector<WeightedEdge> primalEdgesMin;
  std::vector<WeightedEdge> primalEdgesMax;
  std::vector<WeightedEdge> dualEdgesMin;
  std::vector<WeightedEdge> dualEdgesMax;

  /*
  std::vector<WeightedEdge> primalSpanMin;
  std::vector<WeightedEdge> primalSpanMax;
  std::vector<WeightedEdge> dualSpanMin;
  std::vector<WeightedEdge> dualSpanMax;
  */
  std::unordered_set<int> uos;
  for (auto i = 0; i < EV.rows(); i++) {
      Eigen::VectorXd edgeD = V.row(EV(i, 1)) - V.row(EV(i, 0));
      edgeD.normalize();
      EVW.row(i) << abs(edgeD.dot(PD1.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD1.row(EV(i, 1)))) / 2,
          abs(edgeD.dot(PD2.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD2.row(EV(i, 1)))) / 2;
      primalEdgesMin.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 0)));
      primalEdgesMax.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 1)));
      VVE[EV(i, 0)][ EV(i, 1)] = i;
      VVE[EV(i, 1)][ EV(i, 0)] = i;
  }

  for (auto i = 0; i < EF.rows(); i++) {
      //DEV.row(i) << FCV.row(EF(i, 0)), FCV.row(EF(i, 1));// , (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2;
      Eigen::VectorXd edgeD = (FCV.row(EF(i, 1)) - FCV.row(EF(i, 0))) / 2;
      Eigen::VectorXd avgMinC = (PD1.row(F(EV(i, 0), 0)) + PD1.row(F(EV(i, 0), 1)) + PD1.row(F(EV(i, 0), 2)) +
            PD1.row(F(EV(i, 1), 0)) + PD1.row(F(EV(i, 1), 1)) + PD1.row(F(EV(i, 1), 2))) / 3;
      Eigen::VectorXd avgMaxC = (PD2.row(F(EV(i, 0), 0)) + PD2.row(F(EV(i, 0), 1)) + PD2.row(F(EV(i, 0), 2)) +
            PD2.row(F(EV(i, 1), 0)) + PD2.row(F(EV(i, 1), 1)) + PD2.row(F(EV(i, 1), 2))) / 3;
        edgeD.normalize();
        avgMinC.normalize();
        avgMaxC.normalize();

       // DEW.row(i) << abs(edgeD.dot(PD1.row(F(EF(i, 0), 0)))) / 3 + abs(edgeD.dot(PD1.row(F(EF(i, 0), 1)))) / 3 + abs(edgeD.dot(PD1.row(F(EF(i, 0), 2)))) / 3,
            //abs(edgeD.dot(PD2.row(F(EF(i, 1), 0)))) / 3 + abs(edgeD.dot(PD2.row(F(EF(i, 1), 1)))) / 3 + abs(edgeD.dot(PD2.row(F(EF(i, 1), 2)))) / 3;

        //DEW.row(i) << abs(edgeD.dot(avgMinC)), abs(edgeD.dot(avgMaxC));
        //dualEdgesMin.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), DEW(i, 0)));
        //dualEdgesMax.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), DEW(i, 1)));
        dualEdgesMin.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), EVW(i, 0)));
        dualEdgesMax.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), EVW(i, 1)));

      //viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0))+V.row(EV(i,1)))/2, red);
      //viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
  }

  
  int ngc = 0;
  int MAX = 2;// 5 * g;
  int iter = 0;
  bool flag = 0;
  while (goodCycles.size() < 2 * g && iter < MAX) {
      std::vector<std::vector<int>> cycles;
      if (iter % 2 == 0) {
          tree_cotree(primalEdgesMin, dualEdgesMin, cycles,viewer);
          find_good_cycles(cycles);
          visualize_cycles(viewer, cycles, blue);
      }
      else {
          tree_cotree(primalEdgesMax, dualEdgesMax, cycles,viewer);
          find_good_cycles(cycles);
          //visualize_cycles(viewer, cycles, red);
      }
      ++iter;

  }
  cut_graph(goodCycles, viewer);


  //visualize_cycles(viewer, goodCycles, green);

  // Alternative discrete mean curvature
  Eigen::MatrixXd HN;
  Eigen::SparseMatrix<double> L,M,Minv;
  igl::cotmatrix(V,F,L);
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
  igl::invert_diag(M,Minv);
  // Laplace-Beltrami of position
  HN = -Minv*(L*V);
  // Extract magnitude as mean curvature
  Eigen::VectorXd H = HN.rowwise().norm();
  // mean curvature
  H = 0.5*(PV1+PV2);

  //viewer.data().set_data(H);
  // Hide wireframe
  
  viewer.data().show_lines = false;

  viewer.launch();
}
