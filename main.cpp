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
/*
Eigen::MatrixXi TT;
Eigen::MatrixXi TTi;
Eigen::MatrixXi E;
Eigen::MatrixXi uE;
Eigen::MatrixXi EMAP;
std::vector<std::vector<int>> uE2E;
*/
using namespace Eigen;
Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXi EV;
Eigen::MatrixXi FE;
Eigen::MatrixXi EF; //Also dual graph edge vertices in FCV
Eigen::MatrixXd FCV;
Eigen::MatrixXd DEW;
Eigen::MatrixXd EVW;
std::vector<std::vector<int>> goodCycles;
std::vector<int> goodEdgeIndexVec;

const RowVector3d red(0.8, 0.2, 0.2), blue(0.2, 0.2, 0.8);

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

void tree_cotree(std::vector<WeightedEdge> & primalEdges, std::vector<WeightedEdge>& dualEdges, std::vector<std::vector<int>>& cycles) {
    /*
    * Given Primary and Dual edges, the function computes 2*g good cycles.
    */
    std::vector<WeightedEdge>  primalSpan;
    std::vector<WeightedEdge> dualSpan;
    std::vector<std::vector<int>>VE(V.rows(), std::vector<int>());

    std::unordered_set<int> uos;

    //Assign low weight to edges that form good cycles
    for (auto index : goodEdgeIndexVec) {
        primalEdges[index].dist = 0;
    }
    kruskalMST(V.rows(), primalEdges, primalSpan);
    for (auto index : goodEdgeIndexVec) {
        primalEdges[index].dist = primalEdges[index]._dist;
    }
    
    //Add vertex to edge relation for edges in spanning tree
    for (auto edge : primalSpan) {
        uos.insert(edge.index);
        VE[edge.dst].push_back(edge.src);
        VE[edge.src].push_back(edge.dst);
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
    //Add dual graph edges to indexes to set
    for (auto edge : dualSpan) {
        uos.insert(edge.index);
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
void visualize_cycles(igl::opengl::glfw::Viewer &viewer,std::vector<std::vector<int>>& cycles, RowVector3d color) {
    for (auto cycle : cycles) {
        viewer.data().add_edges(V.row(cycle[0]), V.row(cycle.back()), color);
        for (auto j = 1; j < cycle.size(); j++) {
            viewer.data().add_edges(V.row(cycle[j - 1]), V.row(cycle[j]), color);
        }
    }
}

void find_good_cycles() {
    //Find a good cycle and store it in goodCycles


    //Add edge index of the good cycle to goodEdgeIndexVec

}

int main(int argc, char *argv[])
{
  std::string filename = "../resources/fertility.off";
  
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

  
  int g = EV.rows() - V.rows() - F.rows() + 2;
  // Compute curvature directions via quadric fitting
  MatrixXd PD1, PD2;
  VectorXd PV1, PV2;
  igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  const RowVector3d red(0.8, 0.2, 0.2), blue(0.2, 0.2, 0.8);
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
      VectorXd edgeD = V.row(EV(i, 1)) - V.row(EV(i, 0));
      edgeD.normalize();
      EVW.row(i) << abs(edgeD.dot(PD1.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD1.row(EV(i, 1)))) / 2,
          abs(edgeD.dot(PD2.row(EV(i, 0)))) / 2 + abs(edgeD.dot(PD2.row(EV(i, 1)))) / 2;
      primalEdgesMin.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 0)));
      primalEdgesMax.push_back(WeightedEdge(i, EV(i, 0), EV(i, 1), EVW(i, 1)));
  }

  for (auto i = 0; i < EF.rows(); i++) {
      //DEV.row(i) << FCV.row(EF(i, 0)), FCV.row(EF(i, 1));// , (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2;
        VectorXd edgeD = (FCV.row(EF(i, 1)) - FCV.row(EF(i, 0))) / 2;
        VectorXd avgMinC = (PD1.row(F(EV(i, 0), 0)) + PD1.row(F(EV(i, 0), 1)) + PD1.row(F(EV(i, 0), 2)) +
            PD1.row(F(EV(i, 1), 0)) + PD1.row(F(EV(i, 1), 1)) + PD1.row(F(EV(i, 1), 2))) / 3;
        VectorXd avgMaxC = (PD2.row(F(EV(i, 0), 0)) + PD2.row(F(EV(i, 0), 1)) + PD2.row(F(EV(i, 0), 2)) +
            PD2.row(F(EV(i, 1), 0)) + PD2.row(F(EV(i, 1), 1)) + PD2.row(F(EV(i, 1), 2))) / 3;
        edgeD.normalize();
        avgMinC.normalize();
        avgMaxC.normalize();

        DEW.row(i) << abs(edgeD.dot(PD1.row(F(EF(i, 0), 0)))) / 3 + abs(edgeD.dot(PD1.row(F(EF(i, 0), 1)))) / 3 + abs(edgeD.dot(PD1.row(F(EF(i, 0), 2)))) / 3,
            abs(edgeD.dot(PD2.row(F(EF(i, 1), 0)))) / 3 + abs(edgeD.dot(PD2.row(F(EF(i, 1), 1)))) / 3 + abs(edgeD.dot(PD2.row(F(EF(i, 1), 2)))) / 3;

        DEW.row(i) << abs(edgeD.dot(avgMinC)), abs(edgeD.dot(avgMaxC));
        dualEdgesMin.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), DEW(i, 0)));
        dualEdgesMax.push_back(WeightedEdge(i, EF(i, 0), EF(i, 1), DEW(i, 1)));

      //viewer.data().add_edges(FCV.row(EF(i, 0)), (V.row(EV(i, 0))+V.row(EV(i,1)))/2, red);
      //viewer.data().add_edges(FCV.row(EF(i, 1)), (V.row(EV(i, 0)) + V.row(EV(i, 1))) / 2, red);
  }

  
  int ngc = 0;
  int MAX = 2;// 5 * g;
  int iter = 0;
  
  while (ngc < 2 * g && iter < MAX) {
      std::vector<std::vector<int>> cycles;
      if (iter % 2 == 0) {
          tree_cotree(primalEdgesMin, dualEdgesMin, cycles);
          visualize_cycles(viewer, cycles,blue);
      }
      else {
          tree_cotree(primalEdgesMax, dualEdgesMax, cycles);
          visualize_cycles(viewer, cycles,red);
      }
      find_good_cycles();
      ++iter;
  }

  // Alternative discrete mean curvature
  MatrixXd HN;
  SparseMatrix<double> L,M,Minv;
  igl::cotmatrix(V,F,L);
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
  igl::invert_diag(M,Minv);
  // Laplace-Beltrami of position
  HN = -Minv*(L*V);
  // Extract magnitude as mean curvature
  VectorXd H = HN.rowwise().norm();
  // mean curvature
  H = 0.5*(PV1+PV2);

  //viewer.data().set_data(H);
  // Hide wireframe
  
  viewer.data().show_lines = false;

  viewer.launch();
}
