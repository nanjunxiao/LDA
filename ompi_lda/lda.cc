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
/*
  An example running of this program:

  ./lda                                         \
  --num_topics=2                                \
  --alpha=0.1                                   \
  --beta=0.01                                   \
  --training_data_file=./testdata/test_data.txt \
  --model_file=/tmp/lda_model.txt               \
  --burn_in_iterations=100                      \
  --accumulating_iterations=50
*/

#include <time.h>

#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <map>
//added by nanjunxiao
#include <numeric>

#ifdef _OPENMP
#include "omp.h"
#endif // _OPENMP

#include "common.hh"
#include "document.hh"
#include "model.hh"
#include "accumulative_model.hh"
#include "sampler.hh"
#include "cmd_flags.hh"

namespace learning_lda {

  using std::cerr;
  using std::ifstream;
  using std::ofstream;
  using std::istringstream;
  using std::set;
  using std::map;
  using std::string;

  int LoadAndInitTrainingCorpus(const string& corpus_file,
                                int num_topics,
                                LDACorpus* corpus,
                                map<string, int>* word_index_map) {
    corpus->clear();
    word_index_map->clear();
    ifstream fin(corpus_file.c_str());
    string line;
    while (getline(fin, line)) {  // Each line is a training document.
      if (line.size() > 0 &&      // Skip empty lines.
          line[0] != '\r' &&      // Skip empty lines.
          line[0] != '\n' &&      // Skip empty lines.
          line[0] != '#') {       // Skip comment lines.
        istringstream ss(line);
        DocumentWordTopicsPB document;
        string word;
        int count;
        while (ss >> word >> count) {  // Load and init a document.
          vector<int32> topics;
          for (int i = 0; i < count; ++i) {
            topics.push_back(RandInt(num_topics));
          }
          int word_index;
          map<string, int>::const_iterator iter = word_index_map->find(word);
          if (iter == word_index_map->end()) {
            word_index = word_index_map->size();
            (*word_index_map)[word] = word_index;
          } else {
            word_index = iter->second;
          }
          document.add_wordtopics(word, word_index, topics);
        }
        corpus->push_back(new LDADocument(document, num_topics));
      }
    }
    return corpus->size();
  }

  void FreeCorpus(LDACorpus* corpus) {
    for (LDACorpus::iterator iter = corpus->begin();
         iter != corpus->end();
         ++iter) {
      if (*iter != NULL) {
        delete *iter;
        *iter = NULL;
      }
    }
  }

}  // namespace learning_lda

int main(int argc, char** argv) {
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::LDAAccumulativeModel;
  using learning_lda::LDASampler;
  using learning_lda::LDADocument;
  using learning_lda::LoadAndInitTrainingCorpus;
  using learning_lda::LDACmdLineFlags;

  // Command line options
  bool options_parsing_succeeded = false;
  LDACmdLineFlags flags(argc, argv, &options_parsing_succeeded);
  if (!options_parsing_succeeded || !flags.IsValidForTraining()) {
    std::cerr << "Error in parsing command line options.\n";
    return -1;
  }

  // Set random seed
  srand(time(NULL));

#ifdef _OPENMP
  std::cout << "Use " << flags.num_openmp_threads_ << " threads.\n";
  omp_set_num_threads(flags.num_openmp_threads_);
#endif //_OPENMP

  LDACorpus corpus;
  map<string, int> word_index_map;
  CHECK_GT(LoadAndInitTrainingCorpus(flags.training_data_file_,
                                     flags.num_topics_,
                                     &corpus, &word_index_map), 0);

  LDAModel model(flags.num_topics_, word_index_map);
  LDAAccumulativeModel accum_model(flags.num_topics_, word_index_map.size());
  LDASampler sampler(flags.alpha_, flags.beta_, &model, &accum_model);

  sampler.InitializeModel(corpus);

  //modified by nanjunxiao
  vector<TopicProbDistribution> v_prob_dist;//v_prob_dist.reserve(corpus.size() );
  for(int doc=0; doc<corpus.size(); ++doc)
  {
 	TopicProbDistribution prob_dist(flags.num_topics_, 0); 
	v_prob_dist.push_back(prob_dist);//v_prob_dist[doc] = prob_dist;
  }

  //iter
  for (int iter = 0;
       iter < flags.burn_in_iterations_ + flags.accumulating_iterations_;
       ++iter) {
    std::cout << "Iteration " << iter << " ...";
    if (flags.compute_loglikelihood_) {
      std::cout << "log-likelihood: "
                << sampler.ComputeLogLikelihood(corpus) << "\n";
    } else {
      std::cout << "\n";
    }
    //sampler.DoGibbsSampling(&corpus, true, iter < flags.burn_in_iterations_);
    for(int doc=0; doc<corpus.size(); ++doc)
    {
   	sampler.DoGibbsSampling(corpus[doc], true); 
    }//end for doc
    if(iter >= flags.burn_in_iterations_)
    {
	//accumulate model
	accum_model.AccumulateModel(model);
	//accumulate doc's topics_distribution
	for(int doc=0; doc<corpus.size(); ++doc)
	{
	    const vector<int64>& document_distribution = corpus[doc]->topic_distribution();
	    for (int i = 0; i < document_distribution.size(); ++i) 
	    {
            	v_prob_dist[doc][i] += document_distribution[i];
	    }
	}
    }
#ifdef _OPENMP
    //look like useless.
    /*if (iter % flags.correction_period_ == 0) {
      sampler.UpdateModel(corpus);
    }*/
#endif //_OPENMP
  }//end for iter
  accum_model.AverageModel(flags.accumulating_iterations_);
  //added by nanjunxiao
  std::ofstream ftheta(flags.theta_file_.c_str() );
  /*for(int doc=0; doc<corpus.size(); ++doc)
  {
      int sum = std::accumulate( corpus[doc]->topic_distribution().begin(), corpus[doc]->topic_distribution().end(), 0);
      for(int topic=0; topic<corpus[doc]->topic_distribution().size(); ++topic)
      {
	  double theta = ( corpus[doc]->topic_distribution()[topic]+flags.alpha_) / (sum + flags.num_topics_*flags.alpha_ );
	  ftheta <<  theta 
	         << ( (topic<corpus[doc]->topic_distribution().size()-1 ) ? " " : "\n") ;
      }
  }*/
  for(int doc=0; doc<v_prob_dist.size(); ++doc)
  {
      int sum = std::accumulate( v_prob_dist[doc].begin(), v_prob_dist[doc].end(), 0);
      for(int topic=0; topic<v_prob_dist[doc].size(); ++topic)
      {
	  double theta = ( v_prob_dist[doc][topic]/flags.accumulating_iterations_ + flags.alpha_) / (sum/flags.accumulating_iterations_  + flags.num_topics_*flags.alpha_ );
	  ftheta <<  theta 
	         << ( (topic<v_prob_dist[doc].size()-1 ) ? " " : "\n") ;
      }
  }
  ftheta.close();

  FreeCorpus(&corpus);

  std::ofstream fout(flags.model_file_.c_str());
  accum_model.AppendAsString(word_index_map, fout);
  accum_model.SaveModelTWords(word_index_map,flags.twords_file_);
  fout.close();

  return 0;
}
