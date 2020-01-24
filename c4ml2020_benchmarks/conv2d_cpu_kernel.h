/**
  * C4ML/CGO 2020 Submission Benchmark Code
  *
  */

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

namespace Conv2d {

typedef Matrix<float,Dynamic,Dynamic,RowMajor> RMatrix;

class Conv2d {
public:
    Conv2d();

    void Im2Col(const RMatrix& image, RMatrix& data_col, 
                            int height_in, int width_in, int height_kernel, int width_kernel,
                            int height_out, int width_out, int channel_in, int stride_h, int stride_w,
                            int pad_h, int pad_w);
    void Col2Im(const RMatrix& data_col, RMatrix& image,
                            int height_in, int width_in, int height_kernel, int width_kernel,
                            int height_out, int width_out, int channel_in, int stride_h, int stride_w,
                            int pad_h, int pad_w);
    void init();
    void Launch();
    void LaunchBck();
    void LaunchBck2();

    int num_samples;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    int channel_in_;
    int height_in_;
    int width_in_;
    int channel_out_;
    int height_out_;
    int width_out_;
    int height_kernel_;
    int width_kernel_;

    RMatrix input_;
    RMatrix weight_;
    RMatrix bias_;
    RMatrix grad_bias_;
    RMatrix output_;
    RMatrix _input_grad;  //gradient wrt to output

    RMatrix grad_weight_1; //gradient wrt weight Bckw 1
    RMatrix output_grad_1; //gradient wrt to input data Bckw 1
    RMatrix grad_weight_2; //gradient wrt weight Bckw 2
    RMatrix output_grad_2; //gradient wrt to input data Bckw 2

    //std::vector<RMatrix> data_cols_;
};


}


