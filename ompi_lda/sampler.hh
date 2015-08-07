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

#ifndef _OPENSOURCE_GLDA_SAMPLER_H__
#define _OPENSOURCE_GLDA_SAMPLER_H__

#include "common.hh"
#include "document.hh"
#include "model.hh"
#include "accumulative_model.hh"

namespace learning_lda {

// LDASampler trains LDA models and computes statistics about documents in
// LDA models.
class LDASampler {
 public:
  // alpha and beta are the Gibbs sampling symmetric hyperparameters.
  // model is the model to use.
  LDASampler(double alpha, double beta,
             LDAModel* model,
             LDAAccumulativeModel* accum_model);

  ~LDASampler() {}

  // Given a corpus, whose every word has been initialized (i.e.,
  // assigned a random topic), this function initializes model_ to
  // count the word-topic co-occurrences.
  void InitializeModel(const LDACorpus& corpus);

  void UpdateModel(const LDACorpus& corpus);

  // Performs one round of Gibbs sampling on documents in the corpus
  // by invoking DoGibbsSampling(...).  If we are to train
  // a model given training data, we should set update_model to true,
  // and the algorithm updates model_ during Gibbs sampling.
  // Otherwise, if we are to sample the latent topics of a query
  // document, we should set update_model to false.  If update_model is
  // true, burn_in indicates should we accumulate the current estimate
  // to accum_model_.  For the first certain number of iterations,
  // where the algorithm has not converged yet, you should set burn_in
  // to false.  After that, we should set burn_in to true.
  void DoGibbsSampling(LDACorpus* corpus, bool update_model, bool burn_in);

  // Performs one round of Gibbs sampling on a document.  Updates
  // document's topic assignments.  For learning, update_model_=true,
  // for sampling topics of a query, update_model_==false.
  void DoGibbsSampling(LDADocument* document, bool update_model);


  // Computes the log likelihood of a document.
  double ComputeLogLikelihood(const LDACorpus& corpus) const;
  double ComputeLogLikelihood(LDADocument* document) const;

 private:
  const double alpha_;
  const double beta_;
  LDAModel* model_;
  LDAAccumulativeModel* accum_model_;

  // The core of the Gibbs sampling process.  Compute the full conditional
  // posterior distribution of topic assignments to the indicated word.
  //
  // That is, holding all word-topic assignments constant, except for the
  // indicated one, compute a non-normalized probability distribution over
  // topics for the indicated word occurrence.
  void GenerateTopicDistributionForWord(const LDADocument& document,
      int word, int current_word_topic, bool update_model,
      vector<double>* distribution) const;
};


}  // namespace learning_lda

#endif  // _OPENSOURCE_GLDA_SAMPLER_H__
