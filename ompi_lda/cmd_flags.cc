// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cmd_flags.hh"

#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

#include <iostream>
#include <sstream>

#include "boost/program_options/option.hpp"
#include "boost/program_options/options_description.hpp"
#include "boost/program_options/variables_map.hpp"
#include "boost/program_options/parsers.hpp"


namespace learning_lda {

  LDACmdLineFlags::LDACmdLineFlags(int argc, char** argv, bool* succeeded) {
    *succeeded = ParseCmdFlags(argc, argv);
  }

  bool LDACmdLineFlags::ParseCmdFlags(int argc, char** argv) {
    namespace po = boost::program_options;

    po::options_description desc("LDA training options");
    desc.add_options()
      ("num_topics", po::value<int>(&num_topics_)->default_value(0),
       "the number of topics you want to learn from the data")
      ("alpha", po::value<double>(&alpha_)->default_value(0.1),
       "symmetric Dirichlet prior of topics")
      ("beta", po::value<double>(&beta_)->default_value(0.01),
       "symmetric Dirichlet prior of words")
      ("compute_loglikelihood",
       po::value<bool>(&compute_loglikelihood_)->default_value(false),
       "if set to false, do not compute log-likelihood in every training"
       " iteration, so can we save training time.")
      ("training_data_file",
       po::value<std::string>(&training_data_file_)->default_value(""),
       "the text file containing the training data")
      ("inference_data_file",
       po::value<std::string>(&inference_data_file_)->default_value(""),
       "the text file containing the training data")
      ("inference_result_file",
       po::value<std::string>(&inference_result_file_)->default_value(""),
       "the text file containing the training data")
      ("model_file",
       po::value<std::string>(&model_file_)->default_value(""),
       "the text file containing the model (training result)")
      ("twords_file",
       po::value<std::string>(&twords_file_)->default_value(""),
       "the text file containing the model (training result)")
      ("theta_file",
       po::value<std::string>(&theta_file_)->default_value(""),
       "the text file containing the theta ")
      ("burn_in_iterations",
       po::value<int>(&burn_in_iterations_)->default_value(0),
       "the number of iterations used to burn in the MCMC")
      ("accumulating_iterations",
       po::value<int>(&accumulating_iterations_)->default_value(0),
       "the number of iterations after burn_in_iterations to do sampling"
       " and accumulate the sampling result as the training result")
#ifdef _OPENMP
      ("num_openmp_threads",
       po::value<int>(&num_openmp_threads_)->default_value(1),
       "the number of threads that OpenMP starts to do parallel training.")
      ("correction_period",
       po::value<int>(&correction_period_)->default_value(5),
       "the number of iterations after burn_in_iterations to do sampling"
       " and accumulate the sampling result as the training result")
#endif //_OPENMP
      ;

    po::positional_options_description p;
    p.add("training_data_file", -1);
    po::variables_map vm;

    try {
      po::parsed_options parsed =
        po::command_line_parser(argc, argv).options(desc).positional(p).run();
      po::store(parsed, vm);
      po::notify(vm);
    } catch (const std::exception& e) {
      std::cerr << "Bad command line options" << e.what() << "\n";
      return false;
    }
    return true;
  }

  bool LDACmdLineFlags::IsValidForTraining() {
    bool ret = true;
    if (num_topics_ <= 1) {
      std::cerr << "num_topics must >= 2.\n";
      ret = false;
    }
    if (alpha_ <= 0) {
      std::cerr << "alpha must > 0.\n";
      ret = false;
    }
    if (beta_ <= 0) {
      std::cerr << "beta must > 0.\n";
      ret = false;
    }
    if (training_data_file_.empty()) {
      std::cerr << "Invalid training_data_file.\n";
      ret = false;
    }
    if (model_file_.empty()) {
      std::cerr << "Invalid model_file.\n";
      ret = false;
    }
    if (burn_in_iterations_ < 0) {
      std::cerr << "burn_in_iterations must >= 0.\n";
      ret = false;
    }
    if (accumulating_iterations_ <= 0) {
      std::cerr << "accumulating_iterations must > 0.\n";
      ret = false;
    }
#ifdef _OPENMP
    if (correction_period_ <= 0) {
      std::cerr << "correction_period must > 0.\n";
      ret = false;
    }

    if (num_openmp_threads_ <= 0) {
      std::cerr << "num_openmp_threads must > 0\n";
      ret = false;
    }

    if (num_openmp_threads_ > omp_get_max_threads()) {
      std::cerr << "The max num threads on this system is "
                << omp_get_max_threads()
                << ", but you specified " << num_openmp_threads_
                << ". Reset num_openmp_threads to "
                << omp_get_max_threads();
      num_openmp_threads_ = omp_get_max_threads();
      // No error here.
    }
#endif //_OPENMP
    return ret;
  }

  bool LDACmdLineFlags::IsValidForInference() {
    bool ret = true;
    if (alpha_ <= 0) {
      std::cerr << "alpha must > 0.\n";
      ret = false;
    }
    if (beta_ <= 0) {
      std::cerr << "beta must > 0.\n";
      ret = false;
    }
    if (inference_data_file_.empty()) {
      std::cerr << "Invalid inference_data_file.\n";
      ret = false;
    }
    if (inference_result_file_.empty()) {
      std::cerr << "Invalid inference_result_file.\n";
      ret = false;
    }
    if (model_file_.empty()) {
      std::cerr << "Invalid model_file.\n";
      ret = false;
    }
    if (burn_in_iterations_ < 0) {
      std::cerr << "burn_in_iterations must >= 0.\n";
      ret = false;
    }
    if (accumulating_iterations_ <= 0) {
      std::cerr << "accumulating_iterations must > 0.\n";
      ret = false;
    }
    return ret;
  }

}  // namespace learning_lda
