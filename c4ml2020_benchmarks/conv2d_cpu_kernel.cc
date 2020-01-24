/**
 * C4ML/CGO 2020 Submission Benchmark Code
 *
 */
#include "conv2d_cpu_kernel.h"

#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;
using Eigen::TensorMap;
using namespace std;

int main(int argc, char** argv) {
  Conv2d::Conv2d conv;

  conv.init();
  conv.Launch();
  conv.LaunchBck();
  conv.LaunchBck2();
  return 0;
}

namespace Conv2d {

Conv2d::Conv2d()
    : stride_h_(1),
      stride_w_(1),
      pad_h_(0),
      pad_w_(0),
      channel_in_(3),
      height_in_(24),
      width_in_(24),
      channel_out_(6),
      height_out_(12),
      width_out_(12),
      height_kernel_(3),
      width_kernel_(3),
      weight_(),
      bias_(),
      grad_weight_1(),
      grad_bias_(),
      output_(),
      output_grad_1(){};
      //data_cols_(){};

void Conv2d::Conv2d::Im2Col(const RMatrix& image, RMatrix& data_col, 
                            int height_in, int width_in, int height_kernel, int width_kernel,
                            int height_out, int width_out, int channel_in, int stride_h, int stride_w,
                            int pad_h, int pad_w) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;

  #if DEBUG
  std::cout<<"12:"<<hw_out<<" =="<< hw_kernel * channel_in<<std::endl;
  data_col = MatrixXf::Zero(hw_out, hw_kernel * channel_in); //.resize(hw_out, hw_kernel * channel_in);
  data_col.resize(hw_out, hw_kernel * channel_in);
  data_col.setZero();
  double cc = 0;
  clock_t begin;
  #endif

  double i2c_t=0;
  for (int c = 0; c < channel_in; c++) {
    //VectorXf map = image.block(hw_in * c, 0, hw_in, 1);
    for (int i = 0; i < hw_out; i++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride_h + step_w * stride_w;
      if(start_idx / width_in > step_h){
        i = (step_h+1)*width_out -1;
        continue;
      }

      for (int j = 0; j < hw_kernel; j++) {
        // TODO: check the pad_* consistencies
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
          //data_col(i, c * hw_kernel + j) = 0;
        } else {
          int pick_idx = cur_row * width_in + cur_col;
          //data_col(i, c * hw_kernel + j) = map(pick_idx);
          //float IM = image(hw_in * c + pick_idx, 0);//.eval(); //
          //begin = clock();
          data_col(i, c * hw_kernel + j) = image(hw_in * c + pick_idx, 0); //IM; // image(hw_in * c + pick_idx, 0); // = im
          //i2c_t += double(clock() - begin);// / CLOCKS_PER_SEC;
          //cc++;
        }
      }
    }
  }
#if DEBUG
i2c_t /=   CLOCKS_PER_SEC;
std::cout<<"IMG2COL Total:"<<i2c_t<<std::endl;
std::cout<<"Avg:"<<i2c_t/cc<<std::endl;
#endif
}

void Conv2d::Conv2d::Col2Im(const RMatrix& data_col, RMatrix& image,
                            int height_in, int width_in, int height_kernel, int width_kernel,
                            int height_out, int width_out, int channel_in, int stride_h, int stride_w,
                            int pad_h, int pad_w) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  image.resize(hw_in * channel_in, 1);
  image.setZero();
  for (int c = 0; c < channel_in; c++) {
    for (int i = 0; i < hw_out; i++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride_h + step_w * stride_w;
      for (int j = 0; j < hw_kernel; j++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        } else {
          int pick_idx = cur_row * width_in_ + cur_col;
          float IM = data_col(i, c * hw_kernel + j);//.eval();
          image(c * hw_in + pick_idx, 0) += data_col(i, c * hw_kernel + j);//.eval();
        }
      }
    }
  }
}

void Conv2d::Conv2d::init() {

  stride_h_ = 1;
  stride_w_ = 1;
  pad_h_ = 0; 
  pad_w_ = 0;

  num_samples = 32;
  channel_in_ = 320;
  height_in_ = 7;
  width_in_ = 7;
  channel_out_ = 1280;
  height_out_ = 7;
  width_out_ = 7;
  string pad_type = "SAME";
  height_kernel_ = 1;
  width_kernel_ = 1;

  bool has_bias = false;

  // TODO: Check the consistencies with conv2d_op //L2
  if (pad_type == "SAME") {
    pad_h_ /= 2;
    pad_w_ /= 2;
  }

  int dim_in_ = channel_in_ * height_in_ * width_in_;
  input_ = MatrixXf::Random(dim_in_, num_samples).eval();

  int hw_kernel = height_kernel_ * width_kernel_;
  weight_ = MatrixXf::Random(hw_kernel * channel_in_, channel_out_);

  output_.resize(height_out_ * width_out_ * channel_out_, num_samples);
  _input_grad = MatrixXf::Random(height_out_ * width_out_ * channel_out_, num_samples);
    
  grad_weight_1.resize(channel_in_ * height_kernel_ * width_kernel_, channel_out_);
  output_grad_1.resize(channel_in_ * height_in_ * width_in_, num_samples);
  grad_weight_1.setZero();
  output_grad_1.setZero();

  grad_weight_2.resize(channel_in_ * height_kernel_ * width_kernel_, channel_out_); //(channel_out_ * height_kernel_ * width_kernel_, channel_in_);
  output_grad_2.resize(channel_in_ * height_in_ * width_in_, num_samples);
  grad_weight_2.setZero();
  output_grad_2.setZero();
}

void Conv2d::Conv2d::Launch() {
    #if DEBUG
    std::cout<< "Launching Conv2dCpuKernel......"<<std::endl;

    
    input_ = 0.5 * MatrixXf::Ones(dim_in_, num_samples);
    input_.resize(num_samples * channel_in_ * height_in_, width_in_);
    input_ << 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10,
              11, 12, 13, 14, 15,
              16, 17, 18, 19, 20,
              21, 22, 23, 24, 25,
              1, 2, 3, 4, 5,
              6, 7, 8, 9, 10,
              11, 12, 13, 14, 15,
              16, 17, 18, 19, 20,
              21, 22, 23, 24, 25,
              1, 2, 3, 4, 5,
              6, 7, 8, 9, 10,
              11, 12, 13, 14, 15,
              16, 17, 18, 19, 20,
              21, 22, 23, 24, 25,
              1, 2, 3, 4, 5,
              6, 7, 8, 9, 10,
              11, 12, 13, 14, 15,
              16, 17, 18, 19, 20,
              21, 22, 23, 24, 25;

    weight_ = 2 * MatrixXf::Ones(hw_kernel * channel_in_, channel_out_);
    weight_ << 1, 2, 3, 
               4, 5, 6,
               7, 8, 9,
                1, 2, 3, 
               4, 5, 6,
               7, 8, 9,
                1, 2, 3, 
               4, 5, 6,
               7, 8, 9;
    
    data_cols_.resize(num_samples);
    #endif
  
    clock_t begin, end;
    double i2c_t = 0, mm_t = 0;
    for (int i = 0; i < num_samples; i++) {
      RMatrix data_col = MatrixXf::Zero(height_out_ * width_out_, height_kernel_ * width_kernel_ * channel_in_).eval();
     
      //std::cout<<"1:"<<height_out_ * width_out_<<" =="<< height_kernel_ * width_kernel_ * channel_in_<<std::endl;

      begin = clock();
      Im2Col(input_.col(i).eval(), data_col, height_in_, width_in_, height_kernel_, width_kernel_,
                            height_out_, width_out_, channel_in_, stride_h_, stride_w_,
                            pad_h_, pad_w_);
      i2c_t += double(clock() - begin) / CLOCKS_PER_SEC;
      //std::cout<<"Fw data col rows:"<<data_col.rows()<<", "<<data_col.cols()<<std::endl;
      //data_cols_[i] = data_col;
      
      begin = clock();
      RMatrix result = data_col * weight_;
      mm_t += double(clock() - begin) / CLOCKS_PER_SEC;
      
      output_.col(i) = Eigen::Map<VectorXf>(result.data(), result.size());
    }

    std::cout<<std::endl<<"======== Forward Conv Time ========"<<std::endl;
    std::cout<<std::endl<<"I2C (Input): "<<i2c_t<<std::endl;
    std::cout<<"MM (Input, W): "<<mm_t<<std::endl;
    std::cout<<"Total Forward Time: "<<i2c_t <<", "<< mm_t<<std::endl;
    std::cout<<"Total Forward Time: "<<i2c_t + mm_t<<std::endl<<std::endl;
}


typedef Tensor<float, 4, Eigen::RowMajor> tensor4;
typedef TensorMap<tensor4> t_4d;

RMatrix TransposeNC(RMatrix mat, int n, int c, int h, int w) {
  //_  TransposeNC(_input_grad, num_samples, channel_out_, height_out_, width_out_);

  if(h == 1 && w == 1) {
    return mat.transpose();
  }

  RMatrix mat2(n*h*w, c);
  for(int j = 0; j< n; j++) {
    for(int i = 0; i< c; i++) {
      mat2.block(j * h * w, i, h*w, 1) = mat.block(i * h * w, j, h * w, 1);
    }
  }
  return mat2;
  #if DEBUG 
  for(int j = 0; j< n; j++) {
    for(int i = 0; i< c; i++) {
        mat2.col(i).block(j * h * w, 0, h*w, 1) = mat.col(j).block(i * h * w, 0, h * w, 1);
      }
  }
  

  Eigen::array<int, 4> shuffling({3, 1, 2, 0});
  t_4d tens(mat.data(), c, h, w, n);
  tensor4 x = tens.shuffle(shuffling);
  return Eigen::Map<RMatrix>(
            x.data(), n*h*w, c);
  #endif
}

void Conv2d::Conv2d::LaunchBck() {
    

    double trans_wg_t = 0, trans_t = 0, c2i_t = 0, mm_wg_t = 0, mm_ig_t = 0, i2c_t = 0;
    clock_t begin = clock();

    begin = clock();
    RMatrix weight_t = weight_.transpose();
    trans_wg_t += double(clock() - begin) / CLOCKS_PER_SEC;
  
    //GRADIENT wrt INPUT
    for (int i = 0; i < num_samples; i++) {
      RMatrix input_grad_i = _input_grad.col(i);
      RMatrix input_grad_i_col = Eigen::Map<RMatrix>(
          input_grad_i.data(), height_out_ * width_out_ * channel_out_, 1);

      begin = clock();
      input_grad_i_col = TransposeNC(input_grad_i_col, 1, channel_out_, height_out_, width_out_);
      trans_t += double(clock() - begin) / CLOCKS_PER_SEC;

      RMatrix output_grad_i_col(input_grad_i_col.rows(), input_grad_i_col.cols());
      begin = clock();
      output_grad_i_col = (input_grad_i_col * weight_t).eval();
      mm_ig_t += double(clock() - begin) / CLOCKS_PER_SEC;

      RMatrix output_grad_i;
      begin = clock();
      Col2Im(output_grad_i_col, output_grad_i, height_in_, width_in_, height_kernel_, width_kernel_,
                            height_out_, width_out_, channel_in_, stride_h_, stride_w_, pad_h_, pad_w_);
      c2i_t += double(clock() - begin) / CLOCKS_PER_SEC;
      output_grad_1.col(i) = output_grad_i;
    }

    //GRADIENT wrt WEIGHT
    for (int i = 0; i < num_samples; i++) {
      RMatrix input_grad_i = _input_grad.col(i);
      RMatrix input_grad_i_col = Eigen::Map<RMatrix>(
      input_grad_i.data(), height_out_ * width_out_ * channel_out_, 1);

      //begin = clock();
      input_grad_i_col = TransposeNC(input_grad_i_col, 1, channel_out_, height_out_, width_out_);
      //trans_t += double(clock() - begin) / CLOCKS_PER_SEC;

      RMatrix data_col = MatrixXf::Zero(height_out_ * width_out_, height_kernel_ * width_kernel_ * channel_in_).eval();

      begin = clock();
      Im2Col(input_.col(i), data_col, height_in_, width_in_, height_kernel_, width_kernel_,
                            height_out_, width_out_, channel_in_, stride_h_, stride_w_,
                            pad_h_, pad_w_);
      i2c_t += double(clock() - begin) / CLOCKS_PER_SEC;

      //std::cout<<"GEMM BACK1 Input:"<<data_col.cols()<<", "<<data_col.rows()<<std::endl;
      //std::cout<<"HEAD:"<<input_grad_i_col.rows()<<", "<<input_grad_i_col.cols()<<std::endl;

      begin = clock();
      grad_weight_1 += (data_col.transpose() * input_grad_i_col).eval();
      mm_wg_t += double(clock() - begin) / CLOCKS_PER_SEC;
    }
    //std::cout<<std::endl<<"TOTAL RUNNING TIME BACKWARD 1:"<<ad_t<<std::endl;

    std::cout<<std::endl<<"======== Backward wrt Data ========"<<std::endl;
    std::cout<<"HEAD Transpose: "<<trans_t<<std::endl;
    std::cout<<"Weight Transpose: "<<trans_wg_t<<std::endl;
    std::cout<<"MM (in_Grad, TW): "<<mm_ig_t<<std::endl;
    std::cout<<"C2I (mm): "<<c2i_t<<std::endl;
    std::cout<<"Total Backward wrt Data: "<<trans_t <<", "<<trans_wg_t<<", "<< mm_ig_t <<", "<< c2i_t<<std::endl;
    std::cout<<"Total Backward wrt Data: "<<trans_t + trans_wg_t + mm_ig_t + c2i_t <<std::endl<<std::endl;

    std::cout<<std::endl<<"======== Backward wrt Weight ========"<<std::endl;
    std::cout<<"HEAD Transpose: "<<trans_t<<std::endl;
    std::cout<<"I2C (input): "<<i2c_t<<std::endl;
    std::cout<<"MM (input, inGrad): "<<mm_wg_t<<std::endl;
    std::cout<<"Total Backward wrt Weight: "<<trans_t <<", "<< i2c_t <<", "<< mm_wg_t<<std::endl;
    std::cout<<"Total Backward wrt Weight: "<<trans_t + i2c_t + mm_wg_t<<std::endl<<std::endl;

    std::cout<<"Total Backward Time: "<<trans_t + trans_wg_t + mm_ig_t + c2i_t + trans_t + i2c_t + mm_wg_t<<std::endl<<std::endl;
}



void Conv2d::Conv2d::LaunchBck2() {
    
    int h_pad =  (height_kernel_ - 1);
    int w_pad =  (width_kernel_ - 1);
    RMatrix pad_output = RMatrix::Zero(channel_out_ * (height_out_ + (h_pad*2)) * (width_out_ + (w_pad*2)), num_samples);
    clock_t begin;
    double rot_t = 0, pad_t = 0, trans_t = 0, trans_H_t =0, trans_ip_t = 0, trans_wg_t = 0, i2c_H_t = 0, i2c_I_t = 0, mm_wg_t = 0, mm_ig_t = 0;

    begin = clock();
    for(int i = 0; i < channel_out_; i ++) {
      for(int j = (i * (width_out_ + w_pad * 2) * (height_out_ + h_pad*2)) + (h_pad * (width_out_ + (w_pad * 2))), j_grad = 0;
              j_grad < height_out_;
              j += width_out_ + (w_pad * 2), j_grad ++) {
        pad_output.block((j + w_pad), 0, width_out_, num_samples) = _input_grad.block(i * height_out_ * width_out_ + j_grad * width_out_, 0, width_out_, num_samples);
      }
    }
    pad_t += double(clock() - begin) / CLOCKS_PER_SEC;

    RMatrix weight_rev(channel_in_ * height_kernel_ * width_kernel_, channel_out_);
    begin = clock();
    for(int i = 0; i < channel_out_; i++) {
      weight_rev.col(i) = weight_.col(i).reverse();
    }
    rot_t += double(clock() - begin) / CLOCKS_PER_SEC;

    //std::cout<<"W Data:"<<weight_<<std::endl;
    begin = clock();
    weight_rev = TransposeNC(weight_rev, channel_out_, channel_in_, height_kernel_, width_kernel_);
    trans_t += double(clock() - begin) / CLOCKS_PER_SEC;
    //std::cout<<"TransposeNC weight_rev rows,cols:"<<weight_rev.rows()<<", "<<weight_rev.cols()<<std::endl;
    //std::cout<<"TransposeNC weight_rev Data:"<<weight_rev<<std::endl;
    //std::cout<<"transpose weight_rev rows,cols:"<<weight_rev.rows()<<", "<<weight_rev.cols()<<std::endl;
    //std::cout<<"transpose weight_rev Data:"<<weight_rev<<std::endl;
    

    for (int i = 0; i < num_samples; i++) {
        RMatrix data_col = MatrixXf::Zero(height_in_ * width_in_, height_kernel_ * width_kernel_ * channel_out_);
        begin = clock();
        Im2Col(pad_output.col(i), data_col, (height_out_ +  (h_pad*2)), (width_out_ + (w_pad*2)), height_kernel_, width_kernel_,
                              height_in_, width_in_, channel_out_, stride_h_, stride_w_,
                              pad_h_, pad_w_);
        i2c_H_t += double(clock() - begin) / CLOCKS_PER_SEC;

        begin = clock();
        RMatrix result = data_col * weight_rev;
        mm_ig_t += double(clock() - begin) / CLOCKS_PER_SEC;

       for(int j = 0; j < channel_in_; j ++) {
          output_grad_2.block((channel_in_ - j - 1) * (height_in_ * width_in_), i, (height_in_ * width_in_), 1) =
                result.block(0, j, (height_in_ * width_in_), 1);
       }
    }

    //GRADIENT wrt WEIGHT
    //RMatrix input_grad_T = _input_grad.eval();
    RMatrix input_grad_T = Eigen::Map<RMatrix>(
      _input_grad.data(), height_out_ * width_out_ * channel_out_, num_samples);
    begin = clock();
    input_grad_T = TransposeNC(input_grad_T, num_samples, channel_out_, height_out_, width_out_);
    trans_H_t += double(clock() - begin) / CLOCKS_PER_SEC;

    begin = clock();
    RMatrix input_T = TransposeNC(input_, num_samples, channel_in_, height_in_, width_in_);
    trans_ip_t += double(clock() - begin) / CLOCKS_PER_SEC;

    #if DEBUG
    /**Faster version, only one MM **/
    
	RMatrix I2C = MatrixXf::Zero(channel_in_ * height_kernel_ * width_kernel_, height_out_ * width_out_ * num_samples);

	for (int i = 0; i < channel_in_; i++) {
        RMatrix data_col = MatrixXf::Zero(height_kernel_ * width_kernel_, height_out_ * width_out_ * num_samples);
        begin = clock();
        Im2Col(input_T.col(i), data_col, height_in_, width_in_, height_out_, width_out_,
                              height_kernel_, width_kernel_, num_samples, stride_h_, stride_w_,
                              pad_h_, pad_w_);
        std::cout<<"GEMM Input:"<<data_col.rows()<<", "<<data_col.cols()<<std::endl;
        std::cout<<"HEAD:"<<input_grad_T.rows()<<", "<<input_grad_T.cols()<<std::endl;
        std::cout<<"I2C:"<<I2C.rows()<<", "<<I2C.cols()<<std::endl;
        I2C.row(i * height_kernel_ * width_kernel_) = data_col;
        I2C.block(i*height_kernel_*width_kernel_, 0, height_kernel_*width_kernel_, height_out_ * width_out_ * num_samples) = data_col;
        i2c_I_t += double(clock() - begin) / CLOCKS_PER_SEC;

	}
    begin = clock();
    grad_weight_2 = I2C * input_grad_T;
    mm_wg_t += double(clock() - begin) / CLOCKS_PER_SEC;

    std::cout<<"result col rows:"<<result.rows()<<", "<<result.cols()<<std::endl;
    std::cout<<"grad_weight_2 rows:"<<grad_weight_2.rows()<<", "<<grad_weight_2.cols()<<std::endl;
    grad_weight_2 = result;
    #endif 

    for (int i = 0; i < channel_in_; i++) {
        RMatrix data_col = MatrixXf::Zero(height_kernel_ * width_kernel_, height_out_ * width_out_ * num_samples);
        begin = clock();
        Im2Col(input_T.col(i), data_col, height_in_, width_in_, height_out_, width_out_,
                              height_kernel_, width_kernel_, num_samples, stride_h_, stride_w_,
                              pad_h_, pad_w_);
        i2c_I_t += double(clock() - begin) / CLOCKS_PER_SEC;

        //std::cout<<" Input data col rows:"<<data_col.rows()<<", "<<data_col.cols()<<std::endl;
        //std::cout<<" input_grad_T col rows:"<<input_grad_T.rows()<<", "<<input_grad_T.cols()<<std::endl;
        //std::cout<<" grad_weight_2 col rows:"<<grad_weight_2.rows()<<", "<<grad_weight_2.cols()<<std::endl;

        begin = clock();
        RMatrix result = data_col * input_grad_T;
        mm_wg_t += double(clock() - begin) / CLOCKS_PER_SEC;

        if(height_kernel_ == 1 && width_kernel_ ==1)
          grad_weight_2.row(i) = result;
        else  { 
          //for(int j = 0; j < channel_out_; j ++) 
          {
            grad_weight_2.block(i * (height_kernel_ * width_kernel_), 0, (height_kernel_ * width_kernel_), channel_out_) =
                  result.block(0, 0, (height_kernel_ * width_kernel_), channel_out_);
          }
        }
    }

    begin = clock();
    //grad_weight_2 = TransposeNC(grad_weight_2, channel_in_, channel_out_, height_kernel_, width_kernel_);
    trans_wg_t += double(clock() - begin) / CLOCKS_PER_SEC;
    
    std::cout<<std::endl<<"******** Backward2 wrt Data ********"<<std::endl;
    std::cout<<"Padding Time: "<<pad_t<<std::endl;
    std::cout<<"FlipRot Time: "<<rot_t<<std::endl;
    std::cout<<"Weight Transpose: "<<trans_t<<std::endl;
    std::cout<<"I2C (Head): "<<i2c_H_t<<std::endl;
    std::cout<<"MM (Head, W): "<<mm_ig_t<<std::endl;
    std::cout<<"Total Backward wrt Data: "<<pad_t <<", "<< rot_t <<", "<< trans_t <<", "<<  i2c_H_t <<", "<< mm_ig_t<<std::endl;
    std::cout<<"Total Backward wrt Data: "<<pad_t + rot_t + trans_t +  i2c_H_t + mm_ig_t<<std::endl;
    
    std::cout<<std::endl<<"********Backward2 wrt Weight ********"<<std::endl;
    std::cout<<"Head Transpose: "<<trans_H_t<<std::endl;
    std::cout<<"Input Transpose: "<<trans_ip_t<<std::endl;
    std::cout<<"I2C (Input) Time: "<<i2c_I_t<<std::endl;
    std::cout<<"MM (Input, Head) Time: "<<mm_wg_t<<std::endl;
    std::cout<<"WG transpose: "<<trans_wg_t<<std::endl;
    std::cout<<"Total Backward wrt Weight: "<<trans_H_t <<", "<< trans_ip_t <<", "<< i2c_I_t <<", "<< mm_wg_t <<", "<< trans_wg_t<<std::endl;
    std::cout<<"Total Backward wrt Weight: "<<trans_H_t + trans_ip_t + i2c_I_t +  mm_wg_t + trans_wg_t<<std::endl;

    std::cout<<std::endl<<"Total Backward Time: "<<pad_t + rot_t + trans_t +  i2c_H_t + mm_ig_t + trans_H_t + trans_ip_t + i2c_I_t +  mm_wg_t + trans_wg_t<<std::endl;

    std::cout<<std::endl<<"grad wrt input equal:"<<output_grad_1.isApprox(output_grad_2)<<std::endl;
    std::cout<<"grad wrt weight equal:"<<grad_weight_1.isApprox(grad_weight_2)<<std::endl;
}


} // namespace Conv2d
