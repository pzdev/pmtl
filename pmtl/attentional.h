#pragma once

#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dglstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "expr-xtra.h"


#include "sharedGru.h"

#include "relopt-def.h"
#include "dict-utils.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

#define RNN_H0_IS_ZERO

unsigned SLAYERS = 1; // 2
unsigned TLAYERS = 1; // 2
unsigned HIDDEN_DIM = 64; // 1024
unsigned ALIGN_DIM = 32; // 128

unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

int kSRC_SOS;
int kSRC_EOS;
int kSRC_UNK;

int kTGT_SOS;
int kTGT_EOS;
int kTGT_UNK;

//int kSRC_SOS_POS;
//int kSRC_EOS_POS;
//int kSRC_UNK_POS;

using namespace std;

namespace dynet {

    struct ModelStats {
        double loss = 0.0f;
        unsigned words_src = 0;
        unsigned words_tgt = 0;
        unsigned words_src_unk = 0;
        unsigned words_tgt_unk = 0;

        ModelStats() {}
    };



    template<class Builder>
    struct SharedEncoder {
        explicit SharedEncoder(dynet::Model *model, unsigned _src_vocab_size, unsigned slayers,
                               unsigned hidden_dim, LookupParameter *_p_cs = nullptr);

        explicit SharedEncoder(dynet::Model *model, unsigned _src_vocab_size, unsigned slayers, unsigned shared_slayers,
                               unsigned hidden_dim, bool vocab_shared, SharedEncoder<Builder> *p_enc);

        ~SharedEncoder() {}

        LookupParameter p_cs;

        Builder builder_src_fwd;
        Builder builder_src_bwd;
    };


    template<class Builder>
    struct SharedDecoder {
        explicit SharedDecoder(dynet::Model *model, unsigned _tgt_vocab_size, unsigned tlayers, unsigned hidden_dim,
                               unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional, bool _giza_markov,
                               bool _giza_fertility, bool _doc_context, bool _global_fertility, LookupParameter *_p_ct, bool combined_cell_hidden=false);

        explicit SharedDecoder(dynet::Model *model, unsigned _tgt_vocab_size, unsigned tlayers, unsigned shared_tlayers,
                               unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                               bool _giza_markov, bool _giza_fertility, bool _doc_context, bool _global_fertility,
                               bool tvocab_shared, bool align_shared, SharedDecoder<Builder> *p_dec, bool combined_cell_hidden=false);

        ~SharedDecoder() {}

        // s_n: is the representation of the n-th word in the source
        // decoding:
        // e_n = Va * tanh(Wa * h_{t-1} + Ua s_n)
        // \alpha_1..n = softmax(e_1..n)
        // c_t = \sum_n \alpha_n s_n
        // y_t, h_t = buildr(h_{t-1},concat( embedding[w_{t-1}] , c_t ) ) //w_{t-1}: the previous generated target word
        // r_t = R * y_t + b
        // w_t ~ softmax(r_t)

        LookupParameter p_ct;// target vocabulary lookup

        std::vector <Parameter> p_Wh0;

        // context
        Parameter p_S;

        // alignmanet variables
        Parameter p_Wa;
        Parameter p_Ua;
        Parameter p_va;
        Parameter p_Ta;

        // 2-level output softmax
        Parameter p_Q; // used in vanilaLSTM
        Parameter p_P; // used in vanilaLSTM
        Parameter p_R;
        Parameter p_bias;

        Builder builder;
    };

    template<class Builder>
    struct Discriminator {
        explicit Discriminator(dynet::Model *model, unsigned shared_slayers,
                               unsigned hidden_dim, bool _rnn_src_embeddings, unsigned num_tasks);

        ~Discriminator() {}
        // builder traverses shared hidden states of the encoders
        // Then it final state is passes through a non-linear layer followed by a softmax layer
        // r = W * h_t + b
        // p = softmax(r)

        Builder builder;
        Parameter p_W;
        Parameter p_bias;
        unsigned num_tasks;
        unsigned shared_layers;
    };

//===============================================================
//Discriminator =================================================
//===============================================================
//todo:: builder(1, (_rnn_src_embeddings) ? 2 * hidden_dim * shared_slayers : hidden_dim * shared_slayers, hidden_dim, *model)
    template<class Builder>
    Discriminator<Builder>::Discriminator(dynet::Model *model, unsigned shared_slayers,
                                          unsigned hidden_dim, bool _rnn_src_embeddings, unsigned num_tasks)
            : builder(1, 2 * hidden_dim , hidden_dim, *model) {
        //std:://cout<<"builder_adv dim: "<<2*hidden_dim<<"\n";
        p_W = model->add_parameters({num_tasks, hidden_dim});
        p_bias = model->add_parameters({num_tasks});
        this->num_tasks = num_tasks;
        this->shared_layers = shared_slayers;
    }

//===============================================================
//SharedDecoder =================================================
//===============================================================

    // info: creating first decoder (here, we create shared layers)
    template<class Builder>
    SharedDecoder<Builder>::SharedDecoder(dynet::Model *model, unsigned _tgt_vocab_size, unsigned tlayers,
                                          unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings,
                                          bool _giza_positional, bool _giza_markov, bool _giza_fertility,
                                          bool _doc_context, bool _global_fertility, LookupParameter *_p_ct, bool combined_cell_hidden)
            : builder(tlayers, (_rnn_src_embeddings) ? ((combined_cell_hidden)? 5 * hidden_dim : 3 * hidden_dim) : ((combined_cell_hidden)?3 * hidden_dim:2 * hidden_dim), hidden_dim, *model) {

        cout<<"building first model\n";
        p_ct = (_p_ct == nullptr) ? model->add_lookup_parameters(_tgt_vocab_size, {hidden_dim}) : *_p_ct;

        unsigned decoder_coefficient=1;
        if (combined_cell_hidden)
            decoder_coefficient=2; //info: since the return hidden state from the builder is 2*hidden_dim, in the RNN on top of the decoder, we use
        p_R = model->add_parameters({_tgt_vocab_size, hidden_dim*decoder_coefficient});
        p_P = model->add_parameters({decoder_coefficient*hidden_dim, hidden_dim});
        
        p_bias = model->add_parameters({_tgt_vocab_size});
        p_Wa = model->add_parameters({align_dim, tlayers * hidden_dim});

        unsigned coefficient=1;
        if (_rnn_src_embeddings)
            coefficient*=2;
        if (combined_cell_hidden)
            coefficient*=2;
        p_Ua = model->add_parameters({align_dim, coefficient * hidden_dim});
        p_Q = model->add_parameters({decoder_coefficient*hidden_dim, coefficient * hidden_dim});

//        if (_rnn_src_embeddings) {
//            unsigned  coefficient=2;
//            if (combined_cell_hidden)
//                coefficient=4;
//            p_Ua = model->add_parameters({align_dim, coefficient * hidden_dim});
//            p_Q = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            cout<<"coeff: "<< coefficient<<endl;
//            cout<<"p_Q: "<<(coefficient * hidden_dim)<<endl;
//        } else {
//            unsigned coefficient=1;
//            if (combined_cell_hidden)
//                coefficient=2;
//            p_Ua = model->add_parameters({align_dim, coefficient *hidden_dim});
//            p_Q = model->add_parameters({hidden_dim, coefficient *hidden_dim});
//            cout<<"coeff: "<< coefficient<<endl;
//            cout<<"p_Q: "<<(coefficient * hidden_dim)<<endl;
//        }



        if (_giza_positional || _giza_markov || _giza_fertility) {
            int num_giza = 0;
            if (_giza_positional) num_giza += 3;
            if (_giza_markov) num_giza += 3;
            if (_giza_fertility) num_giza += 3;

            p_Ta = model->add_parameters({align_dim, (unsigned int) num_giza});
        }

        p_va = model->add_parameters({align_dim});

        if (_doc_context) {
            p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            if (_rnn_src_embeddings) {
//                unsigned coefficient=2;
//                if (combined_cell_hidden)
//                    coefficient=4;
//                p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            } else {
//                unsigned coefficient=1;
//                if (combined_cell_hidden)
//                    coefficient=2;
//                p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            }
        }

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            if (_rnn_src_embeddings) {
//                unsigned coefficient = 2;
//                if (combined_cell_hidden)
//                    coefficient = 4;
//                p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            }else {
//                unsigned coefficient=2;
//                if (combined_cell_hidden)
//                    coefficient=4;
//                p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            }
        }
    }

    // info: creating other decoders (here, we use shared layers created by the first one)
    template<class Builder>
    SharedDecoder<Builder>::SharedDecoder(dynet::Model *model, unsigned _tgt_vocab_size, unsigned tlayers,
                                          unsigned shared_tlayers, unsigned hidden_dim, unsigned align_dim,
                                          bool _rnn_src_embeddings, bool _giza_positional, bool _giza_markov,
                                          bool _giza_fertility, bool _doc_context, bool _global_fertility,
                                          bool tvocab_shared, bool align_shared, SharedDecoder<Builder> *p_dec, bool combined_cell_hidden)
            : builder(tlayers, shared_tlayers, &(p_dec->builder),
                      (_rnn_src_embeddings) ? ((combined_cell_hidden)? 5 * hidden_dim : 3 * hidden_dim) : ((combined_cell_hidden)? 3 * hidden_dim : 2 * hidden_dim), hidden_dim, *model)
    //	  builder_src_fwd(slayers, shared_slayers, p_enc->builder_src_fwd.params, hidden_dim, hidden_dim, *model)
    {
        cout<<"building other models\n";
        unsigned dec_size=(_rnn_src_embeddings) ? ((combined_cell_hidden)? 5 * hidden_dim : 3 * hidden_dim) : ((combined_cell_hidden)? 3 * hidden_dim : 2 * hidden_dim);
        cout<<"dec input_size: "<<dec_size<<endl;
        if (tvocab_shared) {
            p_ct = p_dec->p_ct;
        } else {
            p_ct = model->add_lookup_parameters(_tgt_vocab_size, {hidden_dim});
        }

        unsigned decoder_coefficient=1;
        if (combined_cell_hidden)
            decoder_coefficient=2; //info: since the return hidden state from the builder is 2*hidden_dim, in the RNN on top of the decoder, we use
        unsigned coefficient=1;
        if (_rnn_src_embeddings)
            coefficient*=2;
        if (combined_cell_hidden)
            coefficient*=2;

        // alignment parameters ----
        if (align_shared) {
            p_Wa = p_dec->p_Wa;
            p_Ua = p_dec->p_Ua;
            p_Ta = p_dec->p_Ta;
            p_va = p_dec->p_va;
        } else {
            p_Wa = model->add_parameters({align_dim, tlayers * hidden_dim});
            p_Ua = model->add_parameters({align_dim, coefficient * hidden_dim});

//            if (_rnn_src_embeddings) {
//                unsigned  coefficient=2;
//                if (combined_cell_hidden)
//                    coefficient=4;
//                p_Ua = model->add_parameters({align_dim, coefficient * hidden_dim});
//            } else {
//                unsigned  coefficient=1;
//                if (combined_cell_hidden)
//                    coefficient=2;
//                p_Ua = model->add_parameters({align_dim, coefficient * hidden_dim});
//            }

            if (_giza_positional || _giza_markov || _giza_fertility) {
                int num_giza = 0;
                if (_giza_positional) num_giza += 3;
                if (_giza_markov) num_giza += 3;
                if (_giza_fertility) num_giza += 3;

                p_Ta = model->add_parameters({align_dim, (unsigned int) num_giza});
            }

            p_va = model->add_parameters({align_dim});
        }

        // context variables ---
        if (_doc_context) {
            p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            if (_rnn_src_embeddings) {
//                unsigned  coefficient=2;
//                if (combined_cell_hidden)
//                    coefficient=4;
//                p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            } else {
//                unsigned  coefficient=1;
//                if (combined_cell_hidden)
//                    coefficient=2;
//                p_S = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            }
        }

        // other params ---
        p_R = model->add_parameters({_tgt_vocab_size, hidden_dim*decoder_coefficient});
        p_P = model->add_parameters({decoder_coefficient*hidden_dim, hidden_dim});
//        p_R = model->add_parameters({_tgt_vocab_size, hidden_dim});
//        p_P = model->add_parameters({hidden_dim, hidden_dim});
        p_bias = model->add_parameters({_tgt_vocab_size});

        p_Q = model->add_parameters({decoder_coefficient*hidden_dim, coefficient * hidden_dim});
        //p_Q = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//        if (_rnn_src_embeddings) {
//            unsigned  coefficient=2;
//            if (combined_cell_hidden)
//                coefficient=4;
//            p_Q = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//        }else {
//            unsigned  coefficient=1;
//            if (combined_cell_hidden)
//                coefficient=2;
//            p_Q = model->add_parameters({hidden_dim, coefficient * hidden_dim});
//            cout<<"coeff: "<< coefficient<<endl;
//            cout<<"p_Q: "<<(coefficient * hidden_dim)<<endl;
//
//        }
        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            if (_rnn_src_embeddings) {
//                unsigned  coefficient=2;
//                if (combined_cell_hidden)
//                    coefficient=4;
//                p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            }else {
//                unsigned  coefficient=1;
//                if (combined_cell_hidden)
//                    coefficient=2;
//                p_Wh0.push_back(model->add_parameters({hidden_dim, coefficient * hidden_dim}));
//            }
        }
    }

//===============================================================
//SharedEncoder =================================================
//===============================================================
    template<class Builder>
    SharedEncoder<Builder>::SharedEncoder(dynet::Model *model, unsigned _src_vocab_size, unsigned slayers,
                                          unsigned hidden_dim, LookupParameter *_p_cs):
            builder_src_fwd(slayers, hidden_dim, hidden_dim, *model),
            builder_src_bwd(slayers, hidden_dim, hidden_dim, *model) {
        p_cs = (_p_cs == nullptr) ? model->add_lookup_parameters(_src_vocab_size, {hidden_dim}) : *_p_cs;
    };

    template<class Builder>
    SharedEncoder<Builder>::SharedEncoder(dynet::Model *model, unsigned _src_vocab_size, unsigned slayers,
                                          unsigned shared_slayers,
                                          unsigned hidden_dim, bool vocab_shared, SharedEncoder<Builder> *p_enc):
            builder_src_fwd(slayers, shared_slayers, &(p_enc->builder_src_fwd), hidden_dim, hidden_dim, *model),
            builder_src_bwd(slayers, shared_slayers, &(p_enc->builder_src_bwd), hidden_dim, hidden_dim, *model) {
        if (vocab_shared)
            p_cs = p_enc->p_cs;
        else
            p_cs = model->add_lookup_parameters(_src_vocab_size, {hidden_dim});
    };


//===============================================================
// AttentionalModel =============================================
//===============================================================

    template<class Builder>
    struct AttentionalModel {
        explicit AttentionalModel(dynet::Model *model,
                                  unsigned _src_vocab_size, unsigned _tgt_vocab_size, unsigned slayers,
                                  unsigned tlayers, unsigned hidden_dim,
                                  unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                                  bool _giza_markov, bool _giza_fertility, bool _doc_context,
                                  bool _global_fertility,
                                  LookupParameter *_p_cs = nullptr, LookupParameter *_p_ct = nullptr);

        explicit AttentionalModel(dynet::Model *model, const SharedEncoder<Builder> *pencoder,
                                  unsigned _src_vocab_size, unsigned _tgt_vocab_size, unsigned slayers,
                                  unsigned tlayers, unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings,
                                  bool _giza_positional, bool _giza_markov, bool _giza_fertility, bool _doc_context,
                                  bool _global_fertility, LookupParameter *_p_ct);

        explicit AttentionalModel(dynet::Model *model, const SharedEncoder<Builder> *pencoder,
                                  SharedDecoder<Builder> *pdecoder,
                                  unsigned _src_vocab_size, unsigned _tgt_vocab_size
                //, unsigned slayers, unsigned tlayers
                , unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                                  bool _giza_markov, bool _giza_fertility, bool _doc_context, bool _global_fertility);

        explicit AttentionalModel(dynet::Model *model, dynet::Model *d_model, Discriminator<Builder> *dist, const SharedEncoder<Builder> *pencoder,
                                  SharedDecoder<Builder> *pdecoder,
                                  unsigned _src_vocab_size, unsigned _tgt_vocab_size
                //, unsigned slayers, unsigned tlayers
                , unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                                  bool _giza_markov, bool _giza_fertility, bool _doc_context, bool _global_fertility);

        ~AttentionalModel();


        vector<bool> check_updated_parameters();
        vector<Expression> AdvLoss_Batch(unsigned current_task, ComputationGraph &cg);

        vector<Expression> AdvLoss(unsigned current_task, ComputationGraph &cg);

        Expression BuildGraph(const std::vector<int> &source, const std::vector<int> &target,
                              ComputationGraph &cg, ModelStats &tstats, Expression *alignment = 0,
                              const std::vector<int> *ctx = 0,
                              Expression *coverage = 0, Expression *fertility = 0);

        void BuildGraph(const std::vector<int> &source, const std::vector<int> &target,
                        ComputationGraph &cg, std::vector <std::vector<float>> &v_preds, bool with_softmax = true);

        Expression
        BuildGraph_Batch(const std::vector <std::vector<int>> &sources, const std::vector <std::vector<int>> &targets,
                         ComputationGraph &cg, ModelStats &tstats, Expression *coverage = 0,
                         Expression *fertility = 0);// for supporting mini-batch training
        Expression
        Forward(const std::vector<int> &sent, int t, bool log_prob, RNNPointer &prev_state, RNNPointer &state,
                dynet::ComputationGraph &cg, std::vector <Expression> &align_out);

        //---------------------------------------------------------------------------------------------
        // Build the relaxation optimization graph for the given sentence including returned loss
        // (Hoang et al., 2017; https://arxiv.org/abs/1701.02854)
        void StartNewInstance(size_t algo, std::vector <dynet::Parameter> &v_params, Dict &sd /*source vocabulary*/
                , ComputationGraph &cg);

        void StartNewInstance(size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params,
                              Dict &sd /*source vocabulary*/
                , ComputationGraph &cg);

        Expression AddInput(const Expression &i_ewe_t, int t, ComputationGraph &cg, RNNPointer *prev_state = 0);

        void ComputeTrgWordEmbeddingMatrix(dynet::ComputationGraph &cg);

        void ComputeSrcWordEmbeddingMatrix(dynet::ComputationGraph &cg);

        Expression GetWordEmbeddingVector(
                const Expression &i_y);

        Expression BuildRelOptGraph(
                size_t algo, std::vector <dynet::Parameter> &v_params /*target*/
                , dynet::ComputationGraph &cg, Dict &d, bool reverse = false,
                Expression *entropy = 0 /*entropy regularization*/
                , Expression *alignment = 0 /*soft alignment*/
                , Expression *coverage = 0, float coverage_C = 1.f /*coverage penalty*/
                , Expression *fertility = 0/*global fertility model*/);

        Expression BuildRelOptGraph(
                size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params /*target*/
                , dynet::ComputationGraph &cg, Dict &d, bool reverse = false,
                Expression *entropy = 0 /*entropy regularization*/
                , Expression *alignment = 0 /*soft alignment*/
                , Expression *coverage = 0, float coverage_C = 1.f /*coverage penalty*/
                , Expression *fertility = 0/*global fertility model*/);

        Expression BuildRevRelOptGraph(
                size_t algo, std::vector <dynet::Parameter> &v_params /*source*/
                , const std::vector<int> &target, dynet::ComputationGraph &cg, Dict &sd, Expression *alignment = 0);

        Expression BuildRevRelOptGraph(
                size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params /*source*/
                , const std::vector<int> &target, dynet::ComputationGraph &cg, Dict &sd, Expression *alignment = 0);

        std::string
        GetRelOptOutput(dynet::ComputationGraph &cg, const std::vector <dynet::Parameter> &v_relopt_params, size_t algo,
                        Dict &d, bool verbose = false);

        std::string GetRelOptOutput(unsigned strategy, dynet::ComputationGraph &cg, const Sentence &i_src_sent,
                                    const std::vector <std::vector<dynet::Parameter>> &v_relopt_params, size_t algo,
                                    Dict &d, bool verbose);

        Expression i_We;// word embedding matrix
        //---------------------------------------------------------------------------------------------

        // enable/disable dropout for source and target RNNs following Gal et al., 2016
        void Set_Dropout(float do_enc, float do_dec);

        void Enable_Dropout();

        void Disable_Dropout();

        void Display_ASCII(const std::vector<int> &source, const std::vector<int> &target,
                           ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td);

        void Display_TIKZ(const std::vector<int> &source, const std::vector<int> &target,
                          ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td);

        void Display_Fertility(const std::vector<int> &source, Dict &sd);

        void Display_Empirical_Fertility(const std::vector<int> &source, const std::vector<int> &target, Dict &sd);

        std::vector<int> Greedy_Decode(const std::vector<int> &source, ComputationGraph &cg,
                                       Dict &tdict, const std::vector<int> *ctx = 0);

        std::vector<int> Beam_Decode(const std::vector<int> &source, ComputationGraph &cg,
                                     unsigned beam_width, Dict &tdict, const std::vector<int> *ctx = 0);

        std::vector<int> Sample(const std::vector<int> &source, ComputationGraph &cg,
                                Dict &tdict, const std::vector<int> *ctx = 0);

        void Add_Global_Fertility_Params(dynet::Model *model, unsigned hidden_dim, bool _rnn_src_embeddings);

        LookupParameter p_cs;// source vocabulary lookup
        LookupParameter p_ct;// target vocabulary lookup
        Parameter p_R;
        Parameter p_Q;
        Parameter p_P;
        Parameter p_S;
        Parameter p_bias;
        Parameter p_Wa;
        std::vector <Parameter> p_Wh0;
        Parameter p_Ua;
        Parameter p_va;
        Parameter p_Ta;
        Parameter p_Wfhid; //fertility parameters
        Parameter p_Wfmu;
        Parameter p_Wfvar;
        Parameter p_bfhid;
        Parameter p_bfmu;
        Parameter p_bfvar;

        Builder builder;
        Builder builder_src_fwd;
        Builder builder_src_bwd;


        bool all_layers_hidden= false;
        bool shared_cells= false;
        bool combined_cell_hidden= false;
        int encoder_layers;
        unsigned shared_layers;
        // info: adversarial parameters
        Builder builder_adv;
        bool is_adversarial = false;
        Parameter p_a_W;
        Parameter p_a_bias;
        unsigned num_tasks;

        bool rnn_src_embeddings;


        bool giza_positional;
        bool giza_markov;
        bool giza_fertility;
        bool doc_context;
        bool global_fertility;



        unsigned src_vocab_size;
        unsigned tgt_vocab_size;

        float dropout_dec;
        float dropout_enc;

        // statefull functions for incrementally creating computation graph, one
        // target word at a time
        void StartNewInstance(const std::vector<int> &src, ComputationGraph &cg, ModelStats &tstats,
                              const std::vector<int> *ctx = 0);

        void StartNewInstance(const std::vector<int> &src, ComputationGraph &cg, const std::vector<int> *ctx = 0);

        void StartNewInstance_Batch(const std::vector <std::vector<int>> &srcs, ComputationGraph &cg,
                                    ModelStats &tstats);// for supporting mini-batch training
        Expression AddInput(unsigned tgt_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state = 0);

        Expression AddInput_Batch(const std::vector<unsigned> &tgt_tok, unsigned t,
                                  ComputationGraph &cg);// for supporting mini-batch training

        std::vector<float> *auxiliary_vector(); // memory management

        // state variables used in the above two methods
        Expression src;
        Expression i_R;
        Expression i_Q;
        Expression i_P;
        Expression i_S;
        Expression i_bias;
        Expression i_Wa;
        Expression i_Ua;
        Expression i_va;
        Expression i_uax;
        Expression i_Ta;
        Expression i_src_idx;
        Expression i_src_len;
        Expression i_tt_ctx;

        Expression i_a_W;
        Expression i_a_bias;

        std::vector <Expression> aligns; // soft word alignments
        unsigned slen; // source sentence length
        bool has_document_context;

        std::vector<std::vector < float>*>
        aux_vecs; // special storage for constant vectors
        unsigned num_aux_vecs;
    };

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression)


    template<class Builder>
    AttentionalModel<Builder>::AttentionalModel(dynet::Model *model, dynet::Model *d_model, Discriminator<Builder> *dist, const SharedEncoder<Builder> *pencoder,
                                                SharedDecoder<Builder> *pdecoder,
                                                unsigned _src_vocab_size, unsigned _tgt_vocab_size
            //, unsigned slayers, unsigned tlayers
            , unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                                                bool _giza_markov, bool _giza_fertility, bool _doc_context,
                                                bool _global_fertility):
            rnn_src_embeddings(_rnn_src_embeddings), giza_positional(_giza_positional), giza_markov(_giza_markov),
            giza_fertility(_giza_fertility), doc_context(_doc_context), global_fertility(_global_fertility),
            src_vocab_size(_src_vocab_size), tgt_vocab_size(_tgt_vocab_size), num_aux_vecs(0) {
        //std::cerr << "Attentionalmodel(" << _src_vocab_size  << " " <<  _tgt_vocab_size  << " " <<  slayers << " " << tlayers << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_positional << ":" << _giza_markov << ":" << _giza_fertility << ":" << _global_fertility << " " <<  _doc_context << ")\n";

        p_cs = pencoder->p_cs;//(_p_cs==nullptr)?model->add_lookup_parameters(src_vocab_size, {hidden_dim}):*_p_cs;
        builder_src_fwd = pencoder->builder_src_fwd;
        builder_src_bwd = pencoder->builder_src_bwd;

        builder = pdecoder->builder;
        p_ct = pdecoder->p_ct; //(_p_ct==nullptr)?model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}):*_p_ct;
        p_R = pdecoder->p_R; //model->add_parameters({tgt_vocab_size, hidden_dim});
        p_P = pdecoder->p_P; //model->add_parameters({hidden_dim, hidden_dim});
        p_bias = pdecoder->p_bias; //model->add_parameters({tgt_vocab_size});
        p_Wa = pdecoder->p_Wa; //model->add_parameters({align_dim, tlayers*hidden_dim});
        p_Ua = pdecoder->p_Ua;
        p_Q = pdecoder->p_Q;
        p_Ta = pdecoder->p_Ta;
        p_va = pdecoder->p_va; //model->add_parameters({align_dim});
        p_S = pdecoder->p_S;
        p_Wh0 = pdecoder->p_Wh0;

        dropout_dec = 0.f;
        dropout_enc = 0.f;


        cout<<"set adv parameters of att model";
        builder_adv = dist->builder;
        p_a_bias = dist->p_bias;
        p_a_W = dist->p_W;
        is_adversarial = true;
        num_tasks = dist->num_tasks;
        shared_layers = dist->shared_layers;
    }


    template<class Builder>
    AttentionalModel<Builder>::AttentionalModel(dynet::Model *model, const SharedEncoder<Builder> *pencoder,
                                                SharedDecoder<Builder> *pdecoder,
                                                unsigned _src_vocab_size, unsigned _tgt_vocab_size
            //, unsigned slayers, unsigned tlayers
            , unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings, bool _giza_positional,
                                                bool _giza_markov, bool _giza_fertility, bool _doc_context,
                                                bool _global_fertility):
            rnn_src_embeddings(_rnn_src_embeddings), giza_positional(_giza_positional), giza_markov(_giza_markov),
            giza_fertility(_giza_fertility), doc_context(_doc_context), global_fertility(_global_fertility),
            src_vocab_size(_src_vocab_size), tgt_vocab_size(_tgt_vocab_size), num_aux_vecs(0) {
        //std::cerr << "Attentionalmodel(" << _src_vocab_size  << " " <<  _tgt_vocab_size  << " " <<  slayers << " " << tlayers << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_positional << ":" << _giza_markov << ":" << _giza_fertility << ":" << _global_fertility << " " <<  _doc_context << ")\n";

        p_cs = pencoder->p_cs;//(_p_cs==nullptr)?model->add_lookup_parameters(src_vocab_size, {hidden_dim}):*_p_cs;
        builder_src_fwd = pencoder->builder_src_fwd;
        builder_src_bwd = pencoder->builder_src_bwd;

        builder = pdecoder->builder;
        p_ct = pdecoder->p_ct; //(_p_ct==nullptr)?model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}):*_p_ct;
        p_R = pdecoder->p_R; //model->add_parameters({tgt_vocab_size, hidden_dim});
        p_P = pdecoder->p_P; //model->add_parameters({hidden_dim, hidden_dim});
        p_bias = pdecoder->p_bias; //model->add_parameters({tgt_vocab_size});
        p_Wa = pdecoder->p_Wa; //model->add_parameters({align_dim, tlayers*hidden_dim});
        p_Ua = pdecoder->p_Ua;
        p_Q = pdecoder->p_Q;
        p_Ta = pdecoder->p_Ta;
        p_va = pdecoder->p_va; //model->add_parameters({align_dim});
        p_S = pdecoder->p_S;
        p_Wh0 = pdecoder->p_Wh0;

        dropout_dec = 0.f;
        dropout_enc = 0.f;
    }


    template<class Builder>
    AttentionalModel<Builder>::AttentionalModel(dynet::Model *model, const SharedEncoder<Builder> *pencoder,
                                                unsigned _src_vocab_size, unsigned _tgt_vocab_size, unsigned slayers,
                                                unsigned tlayers, unsigned hidden_dim, unsigned align_dim,
                                                bool _rnn_src_embeddings, bool _giza_positional, bool _giza_markov,
                                                bool _giza_fertility, bool _doc_context, bool _global_fertility,
                                                LookupParameter *_p_ct)
            : builder(tlayers, (_rnn_src_embeddings) ? 3 * hidden_dim : 2 * hidden_dim, hidden_dim, *model)
            //, builder_src_fwd(slayers, hidden_dim, hidden_dim, *model)
            //, builder_src_bwd(slayers, hidden_dim, hidden_dim, *model)
            , rnn_src_embeddings(_rnn_src_embeddings), giza_positional(_giza_positional), giza_markov(_giza_markov),
              giza_fertility(_giza_fertility), doc_context(_doc_context), global_fertility(_global_fertility),
              src_vocab_size(_src_vocab_size), tgt_vocab_size(_tgt_vocab_size), num_aux_vecs(0) {
        //std::cerr << "Attentionalmodel(" << _src_vocab_size  << " " <<  _tgt_vocab_size  << " " <<  slayers << " " << tlayers << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_positional << ":" << _giza_markov << ":" << _giza_fertility << ":" << _global_fertility << " " <<  _doc_context << ")\n";

        p_cs = pencoder->p_cs;//(_p_cs==nullptr)?model->add_lookup_parameters(src_vocab_size, {hidden_dim}):*_p_cs;
        builder_src_fwd = pencoder->builder_src_fwd;
        builder_src_bwd = pencoder->builder_src_bwd;

        p_ct = (_p_ct == nullptr) ? model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}) : *_p_ct;
        p_R = model->add_parameters({tgt_vocab_size, hidden_dim});
        p_P = model->add_parameters({hidden_dim, hidden_dim});
        p_bias = model->add_parameters({tgt_vocab_size});
        p_Wa = model->add_parameters({align_dim, tlayers * hidden_dim});

        if (rnn_src_embeddings) {
            p_Ua = model->add_parameters({align_dim, 2 * hidden_dim});
            p_Q = model->add_parameters({hidden_dim, 2 * hidden_dim});
        } else {
            p_Ua = model->add_parameters({align_dim, hidden_dim});
            p_Q = model->add_parameters({hidden_dim, hidden_dim});
        }

        if (giza_positional || giza_markov || giza_fertility) {
            int num_giza = 0;
            if (giza_positional) num_giza += 3;
            if (giza_markov) num_giza += 3;
            if (giza_fertility) num_giza += 3;

            p_Ta = model->add_parameters({align_dim, (unsigned int) num_giza});
        }

        p_va = model->add_parameters({align_dim});

        if (doc_context) {
            if (rnn_src_embeddings) {
                p_S = model->add_parameters({hidden_dim, 2 * hidden_dim});
            } else {
                p_S = model->add_parameters({hidden_dim, hidden_dim});
            }
        }

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            if (rnn_src_embeddings)
                p_Wh0.push_back(model->add_parameters({hidden_dim, 2 * hidden_dim}));
            else
                p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
        }

        dropout_dec = 0.f;
        dropout_enc = 0.f;
    }


    template<class Builder>
    AttentionalModel<Builder>::AttentionalModel(dynet::Model *model,
                                                unsigned _src_vocab_size, unsigned _tgt_vocab_size, unsigned slayers,
                                                unsigned tlayers, unsigned hidden_dim, unsigned align_dim,
                                                bool _rnn_src_embeddings, bool _giza_positional, bool _giza_markov,
                                                bool _giza_fertility, bool _doc_context, bool _global_fertility,
                                                LookupParameter *_p_cs, LookupParameter *_p_ct)
            : builder(tlayers, (_rnn_src_embeddings) ? 3 * hidden_dim : 2 * hidden_dim, hidden_dim, *model),
              builder_src_fwd(slayers, hidden_dim, hidden_dim, *model),
              builder_src_bwd(slayers, hidden_dim, hidden_dim, *model), rnn_src_embeddings(_rnn_src_embeddings),
              giza_positional(_giza_positional), giza_markov(_giza_markov), giza_fertility(_giza_fertility),
              doc_context(_doc_context), global_fertility(_global_fertility), src_vocab_size(_src_vocab_size),
              tgt_vocab_size(_tgt_vocab_size), num_aux_vecs(0) {
        //std::cerr << "Attentionalmodel(" << _src_vocab_size  << " " <<  _tgt_vocab_size  << " " <<  slayers << " " << tlayers << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_positional << ":" << _giza_markov << ":" << _giza_fertility << ":" << _global_fertility << " " <<  _doc_context << ")\n";

        p_cs = (_p_cs == nullptr) ? model->add_lookup_parameters(src_vocab_size, {hidden_dim}) : *_p_cs;
        p_ct = (_p_ct == nullptr) ? model->add_lookup_parameters(tgt_vocab_size, {hidden_dim}) : *_p_ct;
        p_R = model->add_parameters({tgt_vocab_size, hidden_dim});
        p_P = model->add_parameters({hidden_dim, hidden_dim});
        p_bias = model->add_parameters({tgt_vocab_size});
        p_Wa = model->add_parameters({align_dim, tlayers * hidden_dim});

        if (rnn_src_embeddings) {
            p_Ua = model->add_parameters({align_dim, 2 * hidden_dim});
            p_Q = model->add_parameters({hidden_dim, 2 * hidden_dim});
        } else {
            p_Ua = model->add_parameters({align_dim, hidden_dim});
            p_Q = model->add_parameters({hidden_dim, hidden_dim});
        }

        if (giza_positional || giza_markov || giza_fertility) {
            int num_giza = 0;
            if (giza_positional) num_giza += 3;
            if (giza_markov) num_giza += 3;
            if (giza_fertility) num_giza += 3;

            p_Ta = model->add_parameters({align_dim, (unsigned int) num_giza});
        }

        p_va = model->add_parameters({align_dim});

        if (doc_context) {
            if (rnn_src_embeddings) {
                p_S = model->add_parameters({hidden_dim, 2 * hidden_dim});
            } else {
                p_S = model->add_parameters({hidden_dim, hidden_dim});
            }
        }

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            if (rnn_src_embeddings)
                p_Wh0.push_back(model->add_parameters({hidden_dim, 2 * hidden_dim}));
            else
                p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
        }

        dropout_dec = 0.f;
        dropout_enc = 0.f;
    }

// enable/disable dropout for source and target RNNs
    template<class Builder>
    void AttentionalModel<Builder>::Set_Dropout(float do_enc, float do_dec) {
        dropout_dec = do_dec;
        dropout_enc = do_enc;
    }

    template<class Builder>
    void AttentionalModel<Builder>::Enable_Dropout() {
        builder.set_dropout(dropout_dec);
        builder_src_fwd.set_dropout(dropout_enc);
        builder_src_bwd.set_dropout(dropout_enc);
    }

    template<class Builder>
    void AttentionalModel<Builder>::Disable_Dropout() {
        builder.disable_dropout();
        builder_src_fwd.disable_dropout();
        builder_src_bwd.disable_dropout();
    }

    template<class Builder>
    void AttentionalModel<Builder>::Add_Global_Fertility_Params(dynet::Model *model, unsigned hidden_dim,
                                                                bool _rnn_src_embeddings) {
        if (global_fertility) {
            if (_rnn_src_embeddings) {
                p_Wfhid = model->add_parameters({hidden_dim, 2 * hidden_dim});
            } else {
                p_Wfhid = model->add_parameters({hidden_dim, hidden_dim});
            }

            p_bfhid = model->add_parameters({hidden_dim});
            p_Wfmu = model->add_parameters({hidden_dim});
            p_bfmu = model->add_parameters({1});
            p_Wfvar = model->add_parameters({hidden_dim});
            p_bfvar = model->add_parameters({1});
        }
    }

    template<class Builder>
    AttentionalModel<Builder>::~AttentionalModel() {
        for (auto v: aux_vecs) delete v;
    }

    template<class Builder>
    std::vector<float> *AttentionalModel<Builder>::auxiliary_vector() {
        while (num_aux_vecs >= aux_vecs.size())
            aux_vecs.push_back(new std::vector<float>());
        // NB, we return the last auxiliary vector, AND increment counter
        return aux_vecs[num_aux_vecs++];
    }

    template<class Builder>
    void AttentionalModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg,
                                                     const std::vector<int> *ctx) {
        //cerr << "StartNewInstance::(1)" << endl;
        slen = source.size();

        size_t max_len = slen;
        size_t max_states = slen;
        if(all_layers_hidden){
            slen  *= encoder_layers; //info : slen is equal to max_states and shows the maximum number of states
            max_states *= encoder_layers;
        }
        //cout<<"max len/enc layers/max_states: "<<max_len<<" "<<encoder_layers<<" "<<max_states<<endl;
        std::vector <Expression> source_embeddings;
        if (!rnn_src_embeddings) {
            for (unsigned s = 0; s < slen; ++s)
                source_embeddings.push_back(lookup(cg, p_cs, source[s]));
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as
            // the representation at each position
            std::vector <Expression> src_fwd(max_states);
            builder_src_fwd.new_graph(cg);
            builder_src_fwd.start_new_sequence();
            for (unsigned l = 0; l < max_len; ++l) {
                //src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                if(!all_layers_hidden) {
                    src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                } else{
                    builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                    vector<Expression> last_h = builder_src_fwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++){
                        src_fwd[(l*encoder_layers) + i]= last_h[i];
                    }
                }
            }
            //cout<<"->here SS 11 \n";
            std::vector <Expression> src_bwd(max_states);
            builder_src_bwd.new_graph(cg);
            builder_src_bwd.start_new_sequence();
            for (int l = max_len - 1; l >= 0; --l) {
                // offset by one position to the right, to catch </s> and generally
                // not duplicate the w_t already captured in src_fwd[t]
                //src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                if(!all_layers_hidden) {
                    src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                } else{
                    builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                    vector<Expression> last_h = builder_src_bwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++) {
                        src_bwd[(l * encoder_layers) + i] = last_h[i];
                    }
                }
            }
            //cout<<"->here SS 22 \n";
            for (unsigned i = 0; i < slen; ++i)
                source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
        }
        src = concatenate_cols(source_embeddings);

        // now for the target sentence
        //cerr << "StartNewInstance::(2)" << endl;
        i_R = parameter(cg, p_R); // hidden -> word rep parameter
        i_Q = parameter(cg, p_Q);
        i_P = parameter(cg, p_P);
        i_bias = parameter(cg, p_bias);  // word bias
        i_Wa = parameter(cg, p_Wa);
        i_Ua = parameter(cg, p_Ua);
        i_va = parameter(cg, p_va);
        i_uax = i_Ua * src;


        // reset aux_vecs counter, allowing the memory to be reused
        num_aux_vecs = 0;

        //cerr << "StartNewInstance::(3)" << endl;

        if (giza_fertility || giza_markov || giza_positional) {
            i_Ta = parameter(cg, p_Ta);
            if (giza_positional) {
                i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
                i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
            }
        }

        aligns.clear();
        aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

        // initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
        std::vector<Expression> h0;
        Expression i_src = average(source_embeddings); // try max instead?

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            Expression i_Wh0 = parameter(cg, p_Wh0[l]);
            h0.push_back(tanh(i_Wh0 * i_src));
        }

        builder.new_graph(cg);
        builder.start_new_sequence(h0);
#else
        builder.new_graph(cg);
        builder.start_new_sequence();
#endif

        // document context; n.b. use "0" context for the first sentence
        if (doc_context && ctx != 0) {
            const std::vector<int> &context = *ctx;

            std::vector <Expression> ctx_embed;
            if (!rnn_src_embeddings) {
                for (unsigned s = 1; s + 1 < context.size(); ++s)
                    ctx_embed.push_back(lookup(cg, p_cs, context[s]));
            } else {
                ctx_embed.resize(context.size() - 1);
                builder_src_fwd.start_new_sequence();
                for (unsigned i = 0; i + 1 < context.size(); ++i)
                    ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
            }

            Expression avg_context = average(source_embeddings);
            i_S = parameter(cg, p_S);
            i_tt_ctx = i_S * avg_context;
            has_document_context = true;
        } else {
            has_document_context = false;
        }
    }

    template<class Builder>
    void AttentionalModel<Builder>::StartNewInstance(const std::vector<int> &source, ComputationGraph &cg,
                                                     ModelStats &tstats, const std::vector<int> *ctx) {
        tstats.words_src += source.size() - 1;

        slen = source.size();

        size_t max_len = slen;
        size_t max_states = slen;
        if(all_layers_hidden){
            slen  *= encoder_layers; //info : slen is equal to max_states and shows the maximum number of states
            max_states *= encoder_layers;
        }
        std::vector <Expression> source_embeddings;
        if (!rnn_src_embeddings) {
            for (unsigned s = 0; s < max_len; ++s) {
                tstats.words_src++;
                if (source[s] == kSRC_UNK) tstats.words_src_unk++;
                source_embeddings.push_back(lookup(cg, p_cs, source[s]));
            }
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as
            // the representation at each position
            std::vector <Expression> src_fwd(max_states);
            builder_src_fwd.new_graph(cg);
            builder_src_fwd.start_new_sequence();
            for (unsigned l = 0; l < max_len; ++l) {
                tstats.words_src++;
                if (source[l] == kSRC_UNK) tstats.words_src_unk++;
                //src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                if(!all_layers_hidden) {
                    src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                } else{
                    builder_src_fwd.add_input(lookup(cg, p_cs, source[l]));
                    vector<Expression> last_h = builder_src_fwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++){
                        src_fwd[(l*encoder_layers) + i]= last_h[i];
                    }
                }
            }

            std::vector <Expression> src_bwd(max_states);
            builder_src_bwd.new_graph(cg);
            builder_src_bwd.start_new_sequence();
            for (int l = max_len - 1; l >= 0; --l) {
                // offset by one position to the right, to catch </s> and generally
                // not duplicate the w_t already captured in src_fwd[t]
                //src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                if(!all_layers_hidden) {
                    src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                } else{
                    builder_src_bwd.add_input(lookup(cg, p_cs, source[l]));
                    vector<Expression> last_h = builder_src_bwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++) {
                        src_bwd[(l * encoder_layers) + i] = last_h[i];
                    }
                }
            }
            for (unsigned i = 0; i < slen; ++i)
                source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
        }
        src = concatenate_cols(source_embeddings);

        //info :adversarial
        if(is_adversarial) {
            //cout<<"->here DD 1 \n";
            builder_adv.new_graph(cg);
            //cout<<"->here DD 2 \n";
            builder_adv.start_new_sequence();
            //cout<<"->here DD 3 \n";
            // info: discriminator network works on the hidden states of the encoder ( since we consider
            // only the hidden state of the last layer which is always shared this method works, otherwise I should use nlayer_back() function).
            //cout<<"dim0 :"<<source_embeddings[0].dim().d[0]<<"\n";
            //cout<<"dim1 :"<<source_embeddings[0].dim().d[1]<<"\n";
//            for (unsigned l = 0; l < slen; ++l) {
//                //cout<<"->here DD 4 \n";
//                builder_adv.add_input(source_embeddings[l]);
//            }
//
            if(!all_layers_hidden){
                for (unsigned l = 0; l < max_len; ++l) {
                    builder_adv.add_input(source_embeddings[l]);
                }
            }else {
                for (unsigned l = 0; l < max_len; ++l) {
                    //cout<<"->here DD 4 \n";
                    for (int i = encoder_layers - shared_layers; i < encoder_layers; i++) {
                        builder_adv.add_input(source_embeddings[l * encoder_layers + i]);
                    }
                }
            }
            //cout<<"->here DD 5 \n";
        }

        // now for the target sentence
        i_R = parameter(cg, p_R); // hidden -> word rep parameter
        i_Q = parameter(cg, p_Q);
        i_P = parameter(cg, p_P);
        i_bias = parameter(cg, p_bias);  // word bias
        i_Wa = parameter(cg, p_Wa);
        i_Ua = parameter(cg, p_Ua);
        i_va = parameter(cg, p_va);
        i_uax = i_Ua * src;

        // reset aux_vecs counter, allowing the memory to be reused
        num_aux_vecs = 0;

        if (giza_fertility || giza_markov || giza_positional) {
            i_Ta = parameter(cg, p_Ta);
            if (giza_positional) {
                i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
                i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
            }
        }

        aligns.clear();
        aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

        // initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
        std::vector<Expression> h0;
        Expression i_src = average(source_embeddings); // try max instead?

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            Expression i_Wh0 = parameter(cg, p_Wh0[l]);
            h0.push_back(tanh(i_Wh0 * i_src));
        }

        builder.new_graph(cg);
        builder.start_new_sequence(h0);
#else
        builder.new_graph(cg);
        builder.start_new_sequence();
#endif

        // document context; n.b. use "0" context for the first sentence
        if (doc_context && ctx != 0) {
            const std::vector<int> &context = *ctx;

            std::vector <Expression> ctx_embed;
            if (!rnn_src_embeddings) {
                for (unsigned s = 1; s + 1 < context.size(); ++s)
                    ctx_embed.push_back(lookup(cg, p_cs, context[s]));
            } else {
                ctx_embed.resize(context.size() - 1);
                builder_src_fwd.start_new_sequence();
                for (unsigned i = 0; i + 1 < context.size(); ++i)
                    ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
            }

            Expression avg_context = average(source_embeddings);
            i_S = parameter(cg, p_S);
            i_tt_ctx = i_S * avg_context;
            has_document_context = true;
        } else {
            has_document_context = false;
        }
    }

    void printDim(Expression x, unsigned dim, string name){
        cout<<name<<" - dim "<<dim<<": "<<x.dim().d[dim]<<endl;
    }

    template<class Builder>
    void AttentionalModel<Builder>::StartNewInstance_Batch(const std::vector <std::vector<int>> &sources,
                                                           ComputationGraph &cg, ModelStats &tstats) {
        // Get the max size
        size_t max_len = sources[0].size();
        for (size_t i = 1; i < sources.size(); i++) max_len = std::max(max_len, sources[i].size());

        slen = max_len;
        //todo: fix this
        encoder_layers = 3;

        size_t max_states = max_len;
        if(all_layers_hidden){
            slen = max_len * encoder_layers;
            max_states *= encoder_layers;
        }
        //cout<<"max len/size: "<<max_len*sources.size()<<endl;

        std::vector<unsigned> words(sources.size());

        std::vector <Expression> source_embeddings;
        //cerr << "(1a) embeddings" << endl;
        //cerr << "rnn_src_embeddings" << rnn_src_embeddings<< endl;
        if (!rnn_src_embeddings) {
            for (unsigned l = 0; l < max_len; l++) {
                for (unsigned bs = 0; bs < sources.size(); ++bs) {
                    words[bs] = (l < sources[bs].size()) ? (unsigned) sources[bs][l] : kSRC_EOS;
                    if (l < sources[bs].size()) {
                        tstats.words_src++;
                        if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
                    }
                }
                source_embeddings.push_back(lookup(cg, p_cs, words));
            }
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as
            // the representation at each position
            //cout<<"->here SS 11 \n";
            std::vector <Expression> src_fwd(max_states);
            //cout<<"->here SS 12 \n";

            builder_src_fwd.new_graph(cg);
            builder_src_fwd.start_new_sequence();
            for (unsigned l = 0; l < max_len; l++) {
                for (unsigned bs = 0; bs < sources.size(); ++bs) {
                    words[bs] = (l < sources[bs].size()) ? (unsigned) sources[bs][l] : kSRC_EOS;
                    if (l < sources[bs].size()) {
                        tstats.words_src++;
                        if (sources[bs][l] == kSRC_UNK) tstats.words_src_unk++;
                    }
                }
                if(!all_layers_hidden) {
                    src_fwd[l] = builder_src_fwd.add_input(lookup(cg, p_cs, words));
                    //printDim(src_fwd[l],0,"src_fwd");
                } else{
                    //cout<<"->here SS 2 \n";
                    builder_src_fwd.add_input(lookup(cg, p_cs, words));
                    vector<Expression> last_h = builder_src_fwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++){
                        src_fwd[(l*encoder_layers) + i]= last_h[i];
                    }
                }
            }
            //cout<<"->here SS 21 \n";
            std::vector <Expression> src_bwd(max_states);

            builder_src_bwd.new_graph(cg);
            builder_src_bwd.start_new_sequence();
            for (int l = max_len - 1; l >= 0; --l) { // int instead of unsigned for negative value of l
                // offset by one position to the right, to catch </s> and generally
                // not duplicate the w_t already captured in src_fwd[t]
                for (unsigned bs = 0; bs < sources.size(); ++bs)
                    words[bs] = ((unsigned) l < sources[bs].size()) ? (unsigned) sources[bs][l] : kSRC_EOS;

                if(!all_layers_hidden) {
                    src_bwd[l] = builder_src_bwd.add_input(lookup(cg, p_cs, words));
                } else{
                    //cout<<"->here SS 4 \n";
                    builder_src_bwd.add_input(lookup(cg, p_cs, words));
                    vector<Expression> last_h = builder_src_bwd.final_h();
                    for (unsigned i =0; i < encoder_layers;i++) {
                        src_bwd[(l * encoder_layers) + i] = last_h[i];
                    }
                }
            }

            //cout<<"->here SS 5 \n";

            for (unsigned l = 0; l < slen; ++l)
                source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[l], src_bwd[l]})));

            //info :adversarial
            if(is_adversarial) {
                //cout<<"->here DD 1 \n";
                builder_adv.new_graph(cg);
                //cout<<"->here DD 2 \n";
                builder_adv.start_new_sequence();
                //cout<<"->here DD 3 \n";
                // info: discriminator network works on the hidden states of the encoder ( since we consider
                // only the hidden state of the last layer which is always shared this method works, otherwise I should use nlayer_back() function).
                //info : it should only consider hidden states of shared layers
                if(!all_layers_hidden){
                    for (unsigned l = 0; l < max_len; ++l) {
                        builder_adv.add_input(source_embeddings[l]);
                    }
                }else {
                    for (unsigned l = 0; l < max_len; ++l) {
                        //cout<<"->here DD 4 \n";
                        for (int i = encoder_layers - shared_layers; i < encoder_layers; i++) {
                            builder_adv.add_input(source_embeddings[l * encoder_layers + i]);
                        }
                    }
                }
                //cout<<"->here DD 5 \n";
            }

        }
        src = concatenate_cols(source_embeddings);

        // now for the target sentence
        i_R = parameter(cg, p_R); // hidden -> word rep parameter
        i_Q = parameter(cg, p_Q);
        i_P = parameter(cg, p_P);
        i_bias = parameter(cg, p_bias);  // word bias
        i_Wa = parameter(cg, p_Wa);
        i_Ua = parameter(cg, p_Ua);
        i_va = parameter(cg, p_va);
//        printDim(i_Ua,0,"i_Ua");
//        printDim(i_Ua,1,"i_Ua");
//        printDim(src,0,"src");
//        printDim(src,1,"src");
        i_uax = i_Ua * src;

        // reset aux_vecs counter, allowing the memory to be reused
        num_aux_vecs = 0;

        //cerr << "(1c) structural biases" << endl;
        if (giza_fertility || giza_markov || giza_positional) {
            i_Ta = parameter(cg, p_Ta);
            if (giza_positional) {
                i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
                i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
            }
        }

        //cerr << "(1d) init alignments" << endl;
        aligns.clear();
        aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

        // initialilse h from global information of the source sentence
        //cerr << "(1e) init builder" << endl;
#ifndef RNN_H0_IS_ZERO
        std::vector<Expression> h0;
        Expression i_src = average(source_embeddings); // try max instead?
        int hidden_layers = builder.num_h0_components();

        for (int l = 0; l < hidden_layers; ++l) {
            Expression i_Wh0 = parameter(cg, p_Wh0[l]);
            h0.push_back(tanh(i_Wh0 * i_src));
        }

        builder.new_graph(cg);
        builder.start_new_sequence(h0);
#else
        builder.new_graph(cg);
        builder.start_new_sequence();
#endif
    }

    template<class Builder>
    Expression
    AttentionalModel<Builder>::AddInput(unsigned trg_tok, unsigned t, ComputationGraph &cg, RNNPointer *prev_state) {
        // alignment input
        Expression i_wah_rep;
        if (t > 0) {
            Expression i_h_tm1;
            if (prev_state)
                i_h_tm1 = concatenate(
                        builder.get_h(*prev_state));// This is required for beam search decoding implementation.
            else
                i_h_tm1 = concatenate(builder.final_h());

            Expression i_wah = i_Wa * i_h_tm1;

            // want numpy style broadcasting, but have to do this manually
            i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
        }

        Expression i_e_t;
        if (giza_markov || giza_fertility || giza_positional) {
            std::vector <Expression> alignment_context;
            if (giza_markov || giza_fertility) {
                if (t > 0) {
                    if (giza_fertility) {
                        auto i_aprev = concatenate_cols(aligns);
                        auto i_asum = sum_cols(i_aprev);
                        auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_asum_pm);
                    }
                    if (giza_markov) {
                        auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_alast_pm);
                    }
                } else {
                    // just 6 repeats of the 0 vector
                    auto zeros = repeat(cg, slen, 0, auxiliary_vector());
                    if (giza_fertility) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                    if (giza_markov) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                }
            }

            if (giza_positional) {
                alignment_context.push_back(i_src_idx);
                alignment_context.push_back(i_src_len);
                auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
                alignment_context.push_back(i_tgt_idx);
            }

            auto i_context = concatenate_cols(alignment_context);

            auto i_e_t_input = i_uax + i_Ta * transpose(i_context);

            if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;
            i_e_t = transpose(tanh(i_e_t_input)) * i_va;
        } else {
            if (t > 0)
                i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
            else
                i_e_t = transpose(tanh(i_uax)) * i_va;
        }

        Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
        aligns.push_back(i_alpha_t);
        Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

        // word input
        Expression i_x_t = lookup(cg, p_ct, trg_tok);
        Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

        // y_t = RNN([x_t, a_t])
        Expression i_y_t;
        if (prev_state)
            i_y_t = builder.add_input(*prev_state, input);
        else
            i_y_t = builder.add_input(input);

        // document context if available
        if (doc_context && has_document_context)
            i_y_t = i_y_t + i_tt_ctx;

#ifndef VANILLA_TARGET_LSTM
        // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
        Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
        Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
        Expression i_r_t = affine_transform({i_bias, i_R, i_y_t});
#endif

        return i_r_t;
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::AddInput_Batch(const std::vector<unsigned> &trg_words, unsigned t,
                                                         ComputationGraph &cg) {
        // alignment input
        Expression i_wah_rep;
        //cout << "im here BB 00\n";
        if (t > 0) {
            Expression i_h_tm1 = concatenate(builder.final_h());
            Expression i_wah = i_Wa * i_h_tm1;

            // want numpy style broadcasting, but have to do this manually
            i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
        }

        //cout << "im here BB 11\n";

        Expression i_e_t;
        if (giza_markov || giza_fertility || giza_positional) {
            std::vector <Expression> alignment_context;
            if (giza_markov || giza_fertility) {
                if (t > 0) {
                    if (giza_fertility) {
                        auto i_aprev = concatenate_cols(aligns);
                        auto i_asum = sum_cols(i_aprev);
                        auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_asum_pm);
                    }
                    if (giza_markov) {
                        auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_alast_pm);
                    }
                } else {
                    // just 6 repeats of the 0 vector
                    auto zeros = repeat(cg, slen, 0, auxiliary_vector());
                    if (giza_fertility) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                    if (giza_markov) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                }
            }

            if (giza_positional) {
                alignment_context.push_back(i_src_idx);
                alignment_context.push_back(i_src_len);
                auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
                alignment_context.push_back(i_tgt_idx);
            }

            auto i_context = concatenate_cols(alignment_context);

            auto i_e_t_input = i_uax + i_Ta * transpose(i_context);

            if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;

            i_e_t = transpose(tanh(i_e_t_input)) * i_va;
        } else {
            //cout << "im here BB 22\n";
            if (t > 0)
                i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
            else
                i_e_t = transpose(tanh(i_uax)) * i_va;
            //cout << "im here BB 33\n";
        }
        //cout << "im here BB 44\n";

        Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
        aligns.push_back(i_alpha_t);
        Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?
        //cout << "im here BB 55\n";

        // target word inputs
        Expression i_x_t = lookup(cg, p_ct, trg_words);
        Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));
        //cout << "im here BB 66\n";

        //printDim(input,0,"input");
        // y_t = RNN([x_t, a_t])
        Expression i_y_t = builder.add_input(input);

        //printDim(i_y_t,0,"i_y_t");

        //cout << "im here BB 77\n";


#ifndef VANILLA_TARGET_LSTM
        //cout << "im here BB 77\n";
        // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
//        printDim(i_Q,0,"i_Q");
//        printDim(i_Q,1,"i_Q");
//        printDim(i_c_t,0,"i_c_t");
//        printDim(i_c_t,1,"i_c_t");
        Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
        Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
        //cout << "im here BB 88\n";
        Expression i_r_t = affine_transform({i_bias, i_R, i_y_t});
#endif
        //cout << "im here BB 99\n";
        return i_r_t;
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
                                                     const std::vector<int> &target, ComputationGraph &cg,
                                                     ModelStats &tstats, Expression *alignment,
                                                     const std::vector<int> *ctx, Expression *coverage,
                                                     Expression *fertility) {
        //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
        StartNewInstance(source, cg, tstats, ctx);

        std::vector <Expression> errs;
        const unsigned tlen = target.size() - 1;
        for (unsigned t = 0; t < tlen; ++t) {
            tstats.words_tgt++;
            if (target[t] == kTGT_UNK) tstats.words_tgt_unk++;

            Expression i_r_t = AddInput(target[t], t, cg);
            Expression i_err = pickneglogsoftmax(i_r_t, target[t + 1]);
            errs.push_back(i_err);
        }

        // save the alignment for later
        if (alignment != 0) {
            // pop off the last alignment column
            *alignment = concatenate_cols(aligns);
        }

        if (coverage != nullptr || fertility != nullptr) {
            Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
            Expression i_totals = sum_cols(i_aligns);
            Expression i_total_trim = pickrange(i_totals, 1, slen - 1);// only care about the non-null entries

            // AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
            if (coverage != nullptr) {
                Expression i_ones = repeat(cg, slen - 2, 1.0f, auxiliary_vector());
                Expression i_penalty = squared_distance(i_total_trim, i_ones);
                *coverage = i_penalty;
            }

            // Contextual fertility model (Cohn et al., 2016)
            if (fertility != nullptr) {
                assert(global_fertility);

                Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
                Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
                Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
                Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));
                Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
                Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

                Expression mu_trim = pickrange(mu, 1, slen - 1);
                Expression var_trim = pickrange(var, 1, slen - 1);

#if 0
                /* log-Normal distribution */
                Expression log_fert = log(i_total_trim);
                Expression delta = log_fert - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
#else
                /* Normal distribution */
                Expression delta = i_total_trim - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta),
                                           2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
                // note that as this is the value of the normal density, the errors
                // are not strictly positive
#endif

                //LOLCAT(transpose(i_total_trim));
                //LOLCAT(transpose(mu_trim));
                //LOLCAT(transpose(var_trim));
                //LOLCAT(transpose(partition + exponent));
                //LOLCAT(exp(transpose(partition + exponent)));
            }
        }

        Expression i_nerr = sum(errs);
        return i_nerr;
    }

    template<class Builder>
    vector<bool> AttentionalModel<Builder>::check_updated_parameters() {
        vector<bool> update_status = {false, false, false};
        update_status[1] = p_Wa.is_updated();
        update_status[2] = p_a_W.is_updated();
        return update_status;
    }

    template<class Builder>
    vector<Expression> AttentionalModel<Builder>::AdvLoss_Batch(unsigned current_task, ComputationGraph &cg) {
        std::vector <Expression> errs;

        Expression last_h = builder_adv.back();
        i_a_W = parameter(cg, p_a_W);
        i_a_bias = parameter(cg, p_a_bias);
        Expression r = i_a_W * last_h + i_a_bias;
        Expression p = softmax(r);

        // equation: log(p(m| S(x,y,\theta_{mtl})) where m is the true label
        Expression adv1 = -log(dynet::expr::pick(p, current_task));
        errs.push_back(sum_batches(adv1));


        // equation: -H(p(m| S(x,y,\theta_{mtl}))
        unsigned i=0;
        Expression adv2 = dynet::expr::pick(p, i)*log(dynet::expr::pick(p, i)+1e-8);
        for (unsigned i=1; i< num_tasks; i++){
            adv2 = adv2 + dynet::expr::pick(p, i)*log(dynet::expr::pick(p, i)+1e-8);
        }
        errs.push_back(sum_batches(adv2));
        return errs;
    }

    template<class Builder>
    vector<Expression> AttentionalModel<Builder>::AdvLoss(unsigned current_task, ComputationGraph &cg) {
        std::vector <Expression> errs;

        Expression last_h = builder_adv.back();
        i_a_W = parameter(cg, p_a_W);
        i_a_bias = parameter(cg, p_a_bias);
        Expression r = i_a_W * last_h + i_a_bias;
        Expression p = softmax(r);

        // equation: log(p(m| S(x,y,\theta_{mtl})) where m is the true label
        Expression adv1 = -log(dynet::expr::pick(p, current_task));
        errs.push_back(adv1);


        // todo: causes nan in dev set!
        //equation: -H(p(m| S(x,y,\theta_{mtl}))
        unsigned i=0;
        Expression adv2 = dynet::expr::pick(p, i)*log(dynet::expr::pick(p, i)+1e-8);
        for (unsigned i=1; i< num_tasks; i++){
            adv2 = adv2 + dynet::expr::pick(p, i)*log(dynet::expr::pick(p, i)+1e-8);
        }
        errs.push_back(adv2);
        return errs;
    }



    template<class Builder>
    Expression AttentionalModel<Builder>::BuildGraph_Batch(const std::vector <std::vector<int>> &sources,
                                                           const std::vector <std::vector<int>> &targets,
                                                           ComputationGraph &cg, ModelStats &tstats,
                                                           Expression *coverage, Expression *fertility) {
        StartNewInstance_Batch(sources, cg, tstats);

        std::vector <Expression> errs;

        const unsigned tlen = targets[0].size() - 1;
        std::vector<unsigned> next_words(targets.size()), words(targets.size());
        //cout << "im here AA 00\n";

        for (unsigned t = 0; t < tlen; ++t) {
            for (size_t bs = 0; bs < targets.size(); bs++) {
                words[bs] = (targets[bs].size() > t) ? (unsigned) targets[bs][t] : kTGT_EOS;
                next_words[bs] = (targets[bs].size() > (t + 1)) ? (unsigned) targets[bs][t + 1] : kTGT_EOS;
                if (targets[bs].size() > t) {
                    tstats.words_tgt++;
                    if (targets[bs][t] == kTGT_UNK) tstats.words_tgt_unk++;
                }
            }

            //cout << "im here AA 11\n";
            Expression i_r_t = AddInput_Batch(words, t, cg);
            //cout << "im here AA 22\n";
            Expression i_err = pickneglogsoftmax(i_r_t, next_words);
            //cout << "im here AA 33\n";
            errs.push_back(i_err);
        }
        //cout << "im here AA 44\n";

        // FIXME: pop off the last alignment column?
        Expression i_alignment = concatenate_cols(aligns);

        if (coverage != nullptr || fertility != nullptr) {
            Expression i_totals = sum_cols(i_alignment);
            Expression i_total_trim = pickrange(i_totals, 1, slen - 1);// only care about the non-null entries

            // AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
            if (coverage != nullptr) {
                Expression i_ones = repeat(cg, slen - 2, 1.0f, auxiliary_vector());
                Expression i_penalty = squared_distance(i_total_trim, i_ones);
                *coverage = sum_batches(i_penalty);
            }

            // Contextual fertility model (Cohn et al., 2016)
            if (fertility != nullptr) {
                assert(global_fertility);

                Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
                Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
                Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
                Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));
                Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
                Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));

                Expression mu_trim = pickrange(mu, 1, slen - 1);
                Expression var_trim = pickrange(var, 1, slen - 1);

#if 0
                /* log-Normal distribution */
                Expression log_fert = log(i_total_trim);
                Expression delta = log_fert - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
#else
                /* Normal distribution */
                Expression delta = i_total_trim - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta),
                                           2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = sum_batches(-sum_cols(transpose(partition + exponent)));
                // note that as this is the value of the normal density, the errors
                // are not strictly positive
#endif

                //LOLCAT(transpose(i_total_trim));
                //LOLCAT(transpose(mu_trim));
                //LOLCAT(transpose(var_trim));
                //LOLCAT(transpose(partition + exponent));
                //LOLCAT(exp(transpose(partition + exponent)));
            }
        }

        //cout << "im here AA 22\n";

        Expression i_nerr = sum_batches(sum(errs));
        return i_nerr;
    }

    template<class Builder>
    void AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
                                               const std::vector<int> &target, ComputationGraph &cg,
                                               std::vector <std::vector<float>> &v_preds,
                                               bool with_softmax) {// v_preds looks like: p(t0) p(t1) p(t2) ... p(</s>). (excluding p(<s>))
        StartNewInstance(source, cg, 0);

        v_preds.clear();
        const unsigned tlen = target.size() - 1;
        for (unsigned t = 0; t < tlen; ++t) {
            Expression i_r_t = AddInput(target[t], t, cg);
            if (with_softmax) {// w/ softmax prediction
                Expression i_softmax = softmax(i_r_t);
                v_preds.push_back(as_vector(cg.get_value(i_softmax.i)));
            } else {// w/o softmax prediction
                v_preds.push_back(as_vector(cg.get_value(i_r_t.i)));
            }
        }
    }

    template<class Builder>
    Expression
    AttentionalModel<Builder>::Forward(const std::vector<int> &sent, int t, bool log_prob, RNNPointer &prev_state,
                                       RNNPointer &state, dynet::ComputationGraph &cg,
                                       std::vector <Expression> &align_out) {
        Expression i_r_t;
        if (state == RNNPointer(-1)) {
            i_r_t = AddInput(sent[t], t, cg);
        } else {
            i_r_t = AddInput(sent[t], t, cg, &prev_state);
        }

        Expression i_softmax = (log_prob) ? log_softmax(i_r_t) : softmax(i_r_t);

        align_out.push_back(aligns.back());

        state = builder.state();

        return i_softmax;
    }

//---------------------------------------------------------------------------------------------
// Build the relaxation optimization graph for the given sentence including returned loss
    template<class Builder>
    void AttentionalModel<Builder>::ComputeTrgWordEmbeddingMatrix(dynet::ComputationGraph &cg) {
        std::vector <Expression> vEs(tgt_vocab_size);
        for (unsigned i = 0; i < tgt_vocab_size; i++)
            vEs[i] = lookup(cg, p_ct, i);//hidden_dim x 1
        i_We = concatenate_cols(vEs);/*hidden_dim x TGT_VOCAB_SIZE*/
    }

    template<class Builder>
    void AttentionalModel<Builder>::ComputeSrcWordEmbeddingMatrix(dynet::ComputationGraph &cg) {
        std::vector <Expression> vEs(src_vocab_size);
        for (unsigned i = 0; i < src_vocab_size; i++)
            vEs[i] = lookup(cg, p_cs, i);//hidden_dim x 1
        i_We = concatenate_cols(vEs);/*hidden_dim x SRC_VOCAB_SIZE*/
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::GetWordEmbeddingVector(const Expression &i_y) {
        // expected embedding
        return (i_We/*hidden_dim x VOCAB_SIZE*/ * i_y/*VOCAB_SIZE x 1*/);//hidden_dim x 1
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::AddInput(const Expression &i_ewe_t, int t, ComputationGraph &cg,
                                                   RNNPointer *prev_state) {
        Expression i_wah_rep;
        if (t > 0) {
            auto i_h_tm1 = concatenate(builder.final_h());
            Expression i_wah = i_Wa * i_h_tm1;
            i_wah_rep = concatenate_cols(
                    std::vector<Expression>(slen, i_wah));// want numpy style broadcasting, but have to do this manually
        }

        Expression i_e_t;
        if (giza_markov || giza_fertility || giza_positional) {
            std::vector <Expression> alignment_context;
            if (giza_markov || giza_fertility) {
                if (t > 0) {
                    if (giza_fertility) {
                        auto i_aprev = concatenate_cols(aligns);
                        auto i_asum = sum_cols(i_aprev);
                        auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_asum_pm);
                    }
                    if (giza_markov) {
                        auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
                        alignment_context.push_back(i_alast_pm);
                    }
                } else {
                    // just 6 repeats of the 0 vector
                    auto zeros = repeat(cg, slen, 0, auxiliary_vector());
                    if (giza_fertility) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                    if (giza_markov) {
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                        alignment_context.push_back(zeros);
                    }
                }
            }
            if (giza_positional) {
                alignment_context.push_back(i_src_idx);
                alignment_context.push_back(i_src_len);
                auto i_tgt_idx = repeat(cg, slen, std::log(1.0 + t), auxiliary_vector());
                alignment_context.push_back(i_tgt_idx);
            }

            auto i_context = concatenate_cols(alignment_context);

            auto i_e_t_input = i_uax + i_Ta * transpose(i_context);

            if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;

            i_e_t = transpose(tanh(i_e_t_input)) * i_va;
        } else {
            if (t > 0)
                i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
            else
                i_e_t = transpose(tanh(i_uax)) * i_va;
        }

        Expression i_alpha_t = softmax(i_e_t); // FIXME: consider summing to less than one?
        aligns.push_back(i_alpha_t);
        Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

        // word input
        Expression i_x_t = i_ewe_t;//lookup(cg, p_ct, trg_tok);
        Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));

        // y_t = RNN([x_t, a_t])
        Expression i_y_t;
        if (prev_state)
            i_y_t = builder.add_input(*prev_state, input);
        else
            i_y_t = builder.add_input(input);

        // document context if available
        if (doc_context && has_document_context)
            i_y_t = i_y_t + i_tt_ctx;

#ifndef VANILLA_TARGET_LSTM
        // Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
        Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
        Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
        Expression i_r_t = affine_transform({i_bias, i_R, i_y_t});
#endif

        return i_r_t;
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::BuildRelOptGraph(
            size_t algo, std::vector <dynet::Parameter> &v_params, dynet::ComputationGraph &cg, Dict &d, bool reverse,
            Expression *entropy, Expression *alignment, Expression *coverage, float coverage_C, Expression *fertility) {
        int tlen = v_params.size();// desired target length (excluding BOS and EOS tokens)
        int ind_bos = d.convert("<s>"), ind_eos = d.convert("</s>");

        //std::cerr << "L*=" << tlen << std::endl;

        // collect expected word embeddings
        std::vector <Expression> i_wes(tlen + 1);
        i_wes[0] = lookup(cg, p_ct, ind_bos);// known BOS embedding
        for (auto t : boost::irange(0, tlen)) {
            auto ct = t;
            if (reverse == true) ct = tlen - t - 1;

            if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                Expression i_p = parameter(cg, v_params[ct]);
                i_wes[t + 1] = GetWordEmbeddingVector(softmax(i_p));
            } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                Expression i_p = parameter(cg, v_params[ct]);
                i_wes[t + 1] = GetWordEmbeddingVector(sparsemax(i_p));
            } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                Expression i_p = parameter(cg, v_params[ct]);
                i_wes[t + 1] = GetWordEmbeddingVector(i_p);
            } else
                assert("Unknown relopt algo! Failed!");
        }

        // simulated generation step
        std::vector <Expression> v_costs, v_ents;
        for (auto t : boost::irange(0, tlen + 1)) {
            //std::cerr << "t:" << t << std::endl;
            Expression i_r_t = AddInput(i_wes[t]/*expected word embedding*/, t, cg);

            // Run the softmax and calculate the cost
            Expression i_cost;
            if (t >= tlen) {// for predicting EOS
                i_cost = pickneglogsoftmax(i_r_t, ind_eos);
            } else {// for predicting others
                Expression i_softmax = softmax(i_r_t);

                auto ct = t;
                if (reverse == true) ct = tlen - t - 1;

                Expression i_y;
                if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                    Expression i_p = parameter(cg, v_params[ct]);
                    i_y = softmax(i_p);
                } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                    Expression i_p = parameter(cg, v_params[ct]);
                    i_y = sparsemax(i_p);
                } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                    i_y = parameter(cg, v_params[ct]);
                } else
                    assert("Unknown inference algo!");

                //i_cost = -log(transpose(i_y) * i_softmax);
                i_cost = -transpose(i_y) * log(i_softmax);//FIXME: use log_softmax(i_r_t) instead, faster?

                // add entropy regularizer to SOFTMAX or SPARSEMAX cost function
                //if (algo == RELOPT_ALGO::SOFTMAX || algo == RELOPT_ALGO::SPARSEMAX){
                //	Expression i_entropy = -transpose(i_y) * log(i_y);
                //	v_ents.push_back(i_entropy);
                //}
            }

            v_costs.push_back(i_cost);
        }

        //if (entropy != 0 && v_ents.size() > 0){
        //	*entropy = sum(v_ents);
        //}

        // save the alignment for later
        if (alignment != 0) {
            // pop off the last alignment column
            *alignment = concatenate_cols(aligns);
        }

        if (coverage != nullptr || fertility != nullptr) {
            Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
            Expression i_totals = sum_cols(i_aligns);
            Expression i_total_trim = pickrange(i_totals, 1, slen - 1);// only care about the non-null entries

            // AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
            if (coverage != nullptr) {
                //Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
                Expression i_ones = repeat(cg, slen - 2, coverage_C, auxiliary_vector());
                Expression i_penalty = squared_distance(i_total_trim, i_ones);
                *coverage = i_penalty;
            }

            // contextual fertility model (Cohn et al., 2016)
            if (fertility != nullptr) {
                assert(global_fertility);

                Expression fbias = concatenate_cols(std::vector<Expression>(slen, const_parameter(cg, p_bfhid)));
                Expression mbias = concatenate(std::vector<Expression>(slen, const_parameter(cg, p_bfmu)));
                Expression vbias = concatenate(std::vector<Expression>(slen, const_parameter(cg, p_bfvar)));
                Expression fhid = tanh(transpose(fbias + const_parameter(cg, p_Wfhid) * src));
                Expression mu = mbias + fhid * const_parameter(cg, p_Wfmu);
                Expression var = exp(vbias + fhid * const_parameter(cg, p_Wfvar));

                Expression mu_trim = pickrange(mu, 1, slen - 1);
                Expression var_trim = pickrange(var, 1, slen - 1);

#if 0
                /* log-Normal distribution */
                Expression log_fert = log(i_total_trim);
                Expression delta = log_fert - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
#else
                /* Normal distribution */
                Expression delta = i_total_trim - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta),
                                           2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
                // note that as this is the value of the normal density, the errors
                // are not strictly positive
#endif

                //LOLCAT(transpose(i_total_trim));
                //LOLCAT(transpose(mu_trim));
                //LOLCAT(transpose(var_trim));
                //LOLCAT(transpose(partition + exponent));
                //LOLCAT(exp(transpose(partition + exponent)));
            }
        }

        Expression i_full_cost = sum(v_costs);
        return i_full_cost;
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::BuildRelOptGraph(
            size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params, dynet::ComputationGraph &cg, Dict &d,
            bool reverse, Expression *entropy, Expression *alignment, Expression *coverage, float coverage_C,
            Expression *fertility) {
        int ind_bos = d.convert("<s>"), ind_eos = d.convert("</s>");

        int tlen = v_params[0].size();// desired target length (excluding BOS and EOS tokens)
        //std::cerr << "L*=" << tlen << std::endl;

        // collect expected word embeddings
        //cerr << "BuildRelOptGraph::(1)" << endl;
        std::vector <Expression> i_wes(tlen + 1);
        i_wes[0] = concatenate_to_batch(std::vector<Expression>(v_params.size(), const_lookup(cg, p_ct,
                                                                                              ind_bos)));// known BOS embedding (batched)
        //cerr << "BuildRelOptGraph::(1a)" << endl;
        for (auto t : boost::irange(0, tlen)) {
            //cerr << "t=" << t << endl;
            auto ct = t;
            if (reverse == true) ct = tlen - t - 1;

            //cerr << "BuildRelOptGraph::(1b)" << endl;
            std::vector <Expression> v_ps;
            for (unsigned bs = 0; bs < v_params.size(); bs++) {
                if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                    Expression i_p = parameter(cg, v_params[bs][ct]);
                    v_ps.push_back(GetWordEmbeddingVector(softmax(i_p)));
                } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                    Expression i_p = parameter(cg, v_params[bs][ct]);
                    v_ps.push_back(GetWordEmbeddingVector(sparsemax(i_p)));
                } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                    Expression i_p = parameter(cg, v_params[bs][ct]);
                    v_ps.push_back(GetWordEmbeddingVector(i_p));
                } else
                    assert("Unknown relopt algo! Failed!");
            }

            i_wes[t + 1] = concatenate_to_batch(v_ps);// batched
        }

        // simulated generation step
        //cerr << "BuildRelOptGraph::(2)" << endl;
        std::vector <Expression> v_costs, v_ents;
        std::vector<unsigned> v_EOSs(v_params.size(), ind_eos);
        for (auto t : boost::irange(0, tlen + 1)) {
            //std::cerr << "t:" << t << std::endl;
            Expression i_r_t = AddInput(i_wes[t]/*"batched" expected word embedding*/, t, cg);

            // Run the softmax and calculate the cost
            Expression i_cost;
            if (t >= tlen) {// for predicting EOS
                i_cost = pickneglogsoftmax(i_r_t, v_EOSs);// batched
            } else {// for predicting others
                Expression i_softmax = softmax(i_r_t);

                auto ct = t;
                if (reverse == true) ct = tlen - t - 1;

                std::vector <Expression> v_ys;
                for (unsigned bs = 0; bs < v_params.size(); bs++) {
                    if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                        Expression i_p = parameter(cg, v_params[bs][ct]);
                        v_ys.push_back(softmax(i_p));
                    } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                        Expression i_p = parameter(cg, v_params[bs][ct]);
                        v_ys.push_back(sparsemax(i_p));
                    } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                        v_ys.push_back(parameter(cg, v_params[bs][ct]));
                    } else
                        assert("Unknown inference algo!");
                }

                Expression i_y = concatenate_to_batch(v_ys);// batched

                //i_cost = -log(transpose(i_y) * i_softmax);
                i_cost = -transpose(i_y) * log(i_softmax);//FIXME: use log_softmax(i_r_t) instead, faster?

                // add entropy regularizer to SOFTMAX or SPARSEMAX cost function
                if (algo == RELOPT_ALGO::SOFTMAX || algo == RELOPT_ALGO::SPARSEMAX) {
                    Expression i_entropy = -transpose(i_y) * log(i_y);
                    v_ents.push_back(i_entropy);
                }
            }

            v_costs.push_back(i_cost);
        }

        if (entropy != 0 && v_ents.size() > 0) {
            *entropy = sum_batches(sum(v_ents));
        }

        // save the alignment for later
        if (alignment != 0) {
            // pop off the last alignment column
            *alignment = concatenate_cols(aligns);
        }

        if (coverage != nullptr || fertility != nullptr) {
            Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
            Expression i_totals = sum_cols(i_aligns);
            Expression i_total_trim = pickrange(i_totals, 1, slen - 1);// only care about the non-null entries

            // AM for computer vision paper (Xu K. et al, 2015) has a penalty over alignment rows deviating from 1
            if (coverage != nullptr) {
                //Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
                Expression i_ones = repeat(cg, slen - 2, coverage_C, auxiliary_vector());
                Expression i_penalty = squared_distance(i_total_trim, i_ones);
                *coverage = sum_batches(i_penalty);
            }

            // contextual fertility model (Cohn et al., 2016)
            if (fertility != nullptr) {
                assert(global_fertility);

                Expression fbias = concatenate_cols(std::vector<Expression>(slen, const_parameter(cg, p_bfhid)));
                Expression mbias = concatenate(std::vector<Expression>(slen, const_parameter(cg, p_bfmu)));
                Expression vbias = concatenate(std::vector<Expression>(slen, const_parameter(cg, p_bfvar)));
                Expression fhid = tanh(transpose(fbias + const_parameter(cg, p_Wfhid) * src));
                Expression mu = mbias + fhid * const_parameter(cg, p_Wfmu);
                Expression var = exp(vbias + fhid * const_parameter(cg, p_Wfvar));

                Expression mu_trim = pickrange(mu, 1, slen - 1);
                Expression var_trim = pickrange(var, 1, slen - 1);

#if 0
                /* log-Normal distribution */
                Expression log_fert = log(i_total_trim);
                Expression delta = log_fert - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta), 2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -log_fert - 0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = -sum_cols(transpose(partition + exponent));
#else
                /* Normal distribution */
                Expression delta = i_total_trim - mu_trim;
                //Expression exponent = cdiv(-cwise_multiply(delta, delta), 2.0f * var_trim);
                Expression exponent = cdiv(-cmult(delta, delta),
                                           2.0f * var_trim);// cmult is a new version of cwise_multiply
                Expression partition = -0.5 * log(2.0f * var_trim * 3.14159265359);
                *fertility = sum_batches(-sum_cols(transpose(partition + exponent)));
                // note that as this is the value of the normal density, the errors
                // are not strictly positive
#endif

                //LOLCAT(transpose(i_total_trim));
                //LOLCAT(transpose(mu_trim));
                //LOLCAT(transpose(var_trim));
                //LOLCAT(transpose(partition + exponent));
                //LOLCAT(exp(transpose(partition + exponent)));
            }
        }

        //cerr << "BuildRelOptGraph::(3)" << endl;
        Expression i_full_cost = sum_batches(sum(v_costs));
        return i_full_cost;
    }

    template<class Builder>
    void AttentionalModel<Builder>::StartNewInstance(size_t algo, std::vector <dynet::Parameter> &v_params, Dict &sd,
                                                     ComputationGraph &cg) {
        std::vector <Expression> exp_wrd_embeddings;
        exp_wrd_embeddings.push_back(lookup(cg, p_cs, sd.convert("<s>")));// BOS
        for (auto t : boost::irange(0, (int) v_params.size())) {
            if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                Expression i_p = parameter(cg, v_params[t]);
                exp_wrd_embeddings.push_back(GetWordEmbeddingVector(softmax(i_p)));
            } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                Expression i_p = parameter(cg, v_params[t]);
                exp_wrd_embeddings.push_back(GetWordEmbeddingVector(sparsemax(i_p)));
            } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                Expression i_p = parameter(cg, v_params[t]);
                exp_wrd_embeddings.push_back(GetWordEmbeddingVector(i_p));
            } else
                assert("Unknown relopt algo! Failed!");
        }
        exp_wrd_embeddings.push_back(const_lookup(cg, p_cs, sd.convert("</s>")));// EOS

        slen = exp_wrd_embeddings.size();//v_params.size() + 2/*BOS and EOS*/;
        std::vector <Expression> source_embeddings;
        if (!rnn_src_embeddings) {
            for (unsigned s = 0; s < slen; ++s)
                source_embeddings.push_back(exp_wrd_embeddings[s]);
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as
            // the representation at each position
            std::vector <Expression> src_fwd(slen);
            builder_src_fwd.new_graph(cg, false);// now fixed parameters
            builder_src_fwd.start_new_sequence();
            for (unsigned i = 0; i < slen; ++i)
                src_fwd[i] = builder_src_fwd.add_input(exp_wrd_embeddings[i]);

            std::vector <Expression> src_bwd(slen);
            builder_src_bwd.new_graph(cg, false);// now fixed parameters
            builder_src_bwd.start_new_sequence();
            for (int i = slen - 1; i >= 0; --i) {
                // offset by one position to the right, to catch </s> and generally
                // not duplicate the w_t already captured in src_fwd[t]
                src_bwd[i] = builder_src_bwd.add_input(exp_wrd_embeddings[i]);
            }

            for (unsigned i = 0; i < slen; ++i)
                source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
        }
        src = concatenate_cols(source_embeddings);

        // now for the target sentence
        i_R = const_parameter(cg, p_R); // hidden -> word rep parameter
        i_Q = const_parameter(cg, p_Q);
        i_P = const_parameter(cg, p_P);
        i_bias = const_parameter(cg, p_bias);  // word bias
        i_Wa = const_parameter(cg, p_Wa);
        i_Ua = const_parameter(cg, p_Ua);
        i_va = const_parameter(cg, p_va);
        i_uax = i_Ua * src;

        // reset aux_vecs counter, allowing the memory to be reused
        num_aux_vecs = 0;

        if (giza_fertility || giza_markov || giza_positional) {
            i_Ta = const_parameter(cg, p_Ta);
            if (giza_positional) {
                i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
                i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
            }
        }

        aligns.clear();
        aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

        // initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
        std::vector<Expression> h0;
        Expression i_src = average(source_embeddings); // try max instead?

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            Expression i_Wh0 = const_parameter(cg, p_Wh0[l]);
            h0.push_back(tanh(i_Wh0 * i_src));
        }

        builder.new_graph(cg, false);// now fixed parameters
        builder.start_new_sequence(h0);
#else
        builder.new_graph(cg, false);// now fixed parameters
        builder.start_new_sequence();
#endif
    }

    template<class Builder>
    void AttentionalModel<Builder>::StartNewInstance(size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params,
                                                     Dict &sd, ComputationGraph &cg) {
        std::vector <Expression> exp_wrd_embeddings;
        exp_wrd_embeddings.push_back(concatenate_to_batch(
                std::vector<Expression>(v_params.size(), const_lookup(cg, p_cs, sd.convert("<s>")))));// BOS
        for (auto t : boost::irange(0, (int) v_params[0].size())) {
            std::vector <Expression> v_ps;
            for (unsigned bs = 0; bs < v_params.size(); bs++) {
                if (algo == RELOPT_ALGO::SOFTMAX) {// SOFTMAX approach
                    Expression i_p = parameter(cg, v_params[bs][t]);
                    v_ps.push_back(GetWordEmbeddingVector(softmax(i_p)));
                } else if (algo == RELOPT_ALGO::SPARSEMAX) {// SPARSEMAX approach
                    Expression i_p = parameter(cg, v_params[bs][t]);
                    v_ps.push_back(GetWordEmbeddingVector(sparsemax(i_p)));
                } else if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG) {// EG or AEG approach
                    Expression i_p = parameter(cg, v_params[bs][t]);
                    v_ps.push_back(GetWordEmbeddingVector(i_p));
                } else
                    assert("Unknown relopt algo! Failed!");
            }
            exp_wrd_embeddings.push_back(concatenate_to_batch(v_ps));
        }
        exp_wrd_embeddings.push_back(concatenate_to_batch(
                std::vector<Expression>(v_params.size(), const_lookup(cg, p_cs, sd.convert("</s>")))));// EOS

        slen = exp_wrd_embeddings.size();//v_params.size() + 2/*BOS and EOS*/;
        std::vector <Expression> source_embeddings;
        if (!rnn_src_embeddings) {
            for (unsigned s = 0; s < slen; ++s)
                source_embeddings.push_back(exp_wrd_embeddings[s]);
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as
            // the representation at each position
            std::vector <Expression> src_fwd(slen);
            builder_src_fwd.new_graph(cg, false);// now fixed parameters
            builder_src_fwd.start_new_sequence();
            for (unsigned i = 0; i < slen; ++i)
                src_fwd[i] = builder_src_fwd.add_input(exp_wrd_embeddings[i]);

            std::vector <Expression> src_bwd(slen);
            builder_src_bwd.new_graph(cg, false);// now fixed parameters
            builder_src_bwd.start_new_sequence();
            for (int i = slen - 1; i >= 0; --i) {
                // offset by one position to the right, to catch </s> and generally
                // not duplicate the w_t already captured in src_fwd[t]
                src_bwd[i] = builder_src_bwd.add_input(exp_wrd_embeddings[i]);
            }

            for (unsigned i = 0; i < slen; ++i)
                source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
        }
        src = concatenate_cols(source_embeddings);

        // now for the target sentence
        i_R = const_parameter(cg, p_R); // hidden -> word rep parameter
        i_Q = const_parameter(cg, p_Q);
        i_P = const_parameter(cg, p_P);
        i_bias = const_parameter(cg, p_bias);  // word bias
        i_Wa = const_parameter(cg, p_Wa);
        i_Ua = const_parameter(cg, p_Ua);
        i_va = const_parameter(cg, p_va);
        i_uax = i_Ua * src;

        // reset aux_vecs counter, allowing the memory to be reused
        num_aux_vecs = 0;

        if (giza_fertility || giza_markov || giza_positional) {
            i_Ta = const_parameter(cg, p_Ta);
            if (giza_positional) {
                i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
                i_src_len = repeat(cg, slen, std::log(1.0 + slen), auxiliary_vector());
            }
        }

        aligns.clear();
        aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

        // initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
        std::vector<Expression> h0;
        Expression i_src = average(source_embeddings); // try max instead?

        int hidden_layers = builder.num_h0_components();
        for (int l = 0; l < hidden_layers; ++l) {
            Expression i_Wh0 = const_parameter(cg, p_Wh0[l]);
            h0.push_back(tanh(i_Wh0 * i_src));
        }

        builder.new_graph(cg, false);// now fixed parameters
        builder.start_new_sequence(h0);
#else
        builder.new_graph(cg, false);// now fixed parameters
        builder.start_new_sequence();
#endif
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::BuildRevRelOptGraph(
            size_t algo, std::vector <dynet::Parameter> &v_params /*source*/
            , const std::vector<int> &target, dynet::ComputationGraph &cg, Dict &sd /*source vocabulary*/
            , Expression *alignment) {
        StartNewInstance(algo, v_params, sd, cg);

        std::vector <Expression> errs;
        const unsigned tlen = target.size() - 1;
        for (unsigned t = 0; t < tlen; ++t) {
            Expression i_r_t = AddInput(target[t], t, cg);
            Expression i_err = pickneglogsoftmax(i_r_t, target[t + 1]);
            errs.push_back(i_err);
        }

        // save the alignment for later
        if (alignment != 0) {
            // pop off the last alignment column
            *alignment = concatenate_cols(aligns);
        }

        Expression i_nerr = sum(errs);
        return i_nerr;
    }

    template<class Builder>
    Expression AttentionalModel<Builder>::BuildRevRelOptGraph(
            size_t algo, std::vector <std::vector<dynet::Parameter>> &v_params /*source*/
            , const std::vector<int> &target, dynet::ComputationGraph &cg, Dict &sd /*source vocabulary*/
            , Expression *alignment) {
        StartNewInstance(algo, v_params, sd, cg);

        std::vector <Expression> errs;
        const unsigned tlen = target.size() - 1;
        for (unsigned t = 0; t < tlen; ++t) {
            Expression i_r_t = AddInput(target[t], t, cg);
            Expression i_err = pickneglogsoftmax(i_r_t, target[t + 1]);
            errs.push_back(i_err);
        }

        // save the alignment for later
        if (alignment != 0) {
            // pop off the last alignment column
            *alignment = concatenate_cols(aligns);
        }

        Expression i_nerr = sum_batches(sum(errs));
        return i_nerr;
    }

    template<class Builder>
    std::string AttentionalModel<Builder>::GetRelOptOutput(dynet::ComputationGraph &cg,
                                                           const std::vector <dynet::Parameter> &v_relopt_params,
                                                           size_t algo, Dict &d, bool verbose) {
        int ind_eos = d.convert("</s>");

        std::stringstream ss;
        for (auto &p : v_relopt_params) {
            Expression i_y;
            if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG)
                i_y = parameter(cg, p);
            else if (algo == RELOPT_ALGO::SOFTMAX)
                i_y = softmax(parameter(cg, p));
            else if (algo == RELOPT_ALGO::SPARSEMAX)
                i_y = sparsemax(parameter(cg, p));

            //cg.incremental_forward();

            std::vector<float> v_y_dist = as_vector(i_y.value());

            // FIXME: Add the bos/eos/unk penalties if required
            //v_y_dist[] = -1.f;// penalty since <s> never appears in the middle of a target sentence.
            //v_y_dist[GetUnkId()] = -1.f;

            //cerr << "[y]=" << print_vec(v_y_dist) << endl;
            std::vector<float>::iterator it = std::max_element(v_y_dist.begin(), v_y_dist.end());
            int index = std::distance(v_y_dist.begin(), it);
            if (index == ind_eos)// FIXME: early ignorance
                break;
            ss << d.convert(index) << " ";
            if (verbose) std::cerr << d.convert(index) << "(" << *it << ")" << " ";// console output
        }

        if (verbose) std::cerr << std::endl;

        return ss.str();// optimised output
    }

    template<class Builder>
    std::string AttentionalModel<Builder>::GetRelOptOutput(unsigned strategy, dynet::ComputationGraph &cg,
                                                           const Sentence &i_src_sent,
                                                           const std::vector <std::vector<dynet::Parameter>> &v_relopt_params,
                                                           size_t algo, Dict &d, bool verbose) {
        int ind_bos = d.convert("<s>"), ind_eos = d.convert("</s>");

        Sentence i_best_dec_sent;
        float best_loss = std::numeric_limits<float>::max();
        if (strategy == 0) {//minimum loss strategy
            for (unsigned k = 0; k < v_relopt_params.size(); k++) {
                Sentence i_trg;
                for (auto &p : v_relopt_params[k]) {
                    Expression i_y;
                    if (algo == RELOPT_ALGO::EG || algo == RELOPT_ALGO::AEG)
                        i_y = parameter(cg, p);
                    else if (algo == RELOPT_ALGO::SOFTMAX)
                        i_y = softmax(parameter(cg, p));
                    else if (algo == RELOPT_ALGO::SPARSEMAX)
                        i_y = sparsemax(parameter(cg, p));

                    //cg.incremental_forward();

                    std::vector<float> v_y_dist = as_vector(i_y.value());

                    // FIXME: Add the bos/eos/unk penalties if required
                    //v_y_dist[] = -1.f;// penalty since <s> never appears in the middle of a target sentence.
                    //v_y_dist[GetUnkId()] = -1.f;

                    //cerr << "[y]=" << print_vec(v_y_dist) << endl;
                    std::vector<float>::iterator it = std::max_element(v_y_dist.begin(), v_y_dist.end());
                    int index = std::distance(v_y_dist.begin(), it);
                    if (index == ind_eos)// FIXME: early ignorance
                        break;
                    i_trg.push_back(index);
                    //ss << d.convert(index) << " ";
                    //if (verbose) std::cerr << d.convert(index) << "(" << *it << ")" << " ";// console output
                }

                // get the loss --> FIXME: take time?
                i_trg.insert(i_trg.begin(), ind_bos);// BOS
                i_trg.push_back(ind_eos);// EOS
                ModelStats stats;
                auto iloss =
                        (1.f / (float) (i_trg.size() - 1)) * BuildGraph(i_src_sent, i_trg, cg, stats);// normalized NLL
                float loss = as_scalar(cg.forward(iloss));
                cerr << "loss_" << k << "=" << loss << endl;
                cerr << "sent_" << k << "=" << Convert2Str(d, i_trg, false) << endl;
                if (loss < best_loss) {
                    best_loss = loss;
                    i_best_dec_sent = i_trg;
                }
            }
        } else if (strategy == 1) {//average distribution strategy
            // FIXME
        }

        if (verbose) std::cerr << std::endl;

        //cerr << "best_loss=" << best_loss << endl;

        return Convert2Str(d, i_best_dec_sent, false);// optimised output
    }
//---------------------------------------------------------------------------------------------

    template<class Builder>
    void AttentionalModel<Builder>::Display_ASCII(const std::vector<int> &source, const std::vector<int> &target,
                                                  ComputationGraph &cg, const Expression &alignment, Dict &sd,
                                                  Dict &td) {
        using namespace std;

        // display the alignment
        //float I = target.size() - 1;
        //float J = source.size() - 1;
        unsigned I = target.size();
        unsigned J = source.size();
        //vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
        vector <string> symbols{".", "o", "*", "O", "@"};
        int num_symbols = symbols.size();
        vector<float> thresholds;
        thresholds.push_back(0.8 / I);
        float lgap = (0 - std::log(thresholds.back())) / (num_symbols - 1);
        for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
            float thr = std::exp(std::log(thresholds.back()) + lgap);
            thresholds.push_back(thr);
        }
        // FIXME: thresholds > 1, what's going on?
        //cout << thresholds.back() << endl;

        const Tensor &a = cg.get_value(alignment.i);
        //cout << "I = " << I << " J = " << J << endl;

        cout.setf(ios_base::adjustfield, ios_base::left);
        cout << setw(12) << "source" << "  ";
        cout.setf(ios_base::adjustfield, ios_base::right);
        for (unsigned j = 0; j < J; ++j)
            cout << setw(2) << j << ' ';
        cout << endl;

        for (unsigned i = 0; i < I; ++i) {
            cout.setf(ios_base::adjustfield, ios_base::left);
            //cout << setw(12) << td.convert(target[i+1]) << "  ";
            cout << setw(12) << td.convert(target[i]) << "  ";
            cout.setf(ios_base::adjustfield, ios_base::right);
            float max_v = 0;
            int max_j = -1;
            for (unsigned j = 0; j < J; ++j) {
                float v = TensorTools::access_element(a, Dim({(unsigned int) j, (unsigned int) i}));
                string symbol;
                for (int s = 0; s <= num_symbols; ++s) {
                    if (s == 0)
                        symbol = ' ';
                    else
                        symbol = symbols[s - 1];
                    if (s != num_symbols && v < thresholds[s])
                        break;
                }
                cout << setw(2) << symbol << ' ';
                if (v >= max_v) {
                    max_v = v;
                    max_j = j;
                }
            }
            cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
        }
        cout << resetiosflags(ios_base::adjustfield);
        for (unsigned j = 0; j < J; ++j)
            cout << j << ":" << sd.convert(source[j]) << ' ';
        cout << endl;
    }

    template<class Builder>
    void AttentionalModel<Builder>::Display_TIKZ(const std::vector<int> &source, const std::vector<int> &target,
                                                 ComputationGraph &cg, const Expression &alignment, Dict &sd,
                                                 Dict &td) {
        using namespace std;

        // display the alignment
        unsigned I = target.size();
        unsigned J = source.size();

        const Tensor &a = cg.get_value(alignment.i);
        cout << a.d[0] << " x " << a.d[1] << endl;

        cout << "\\begin{tikzpicture}[scale=0.5]\n";
        for (unsigned j = 0; j < J; ++j)
            cout << "\\node[anchor=west,rotate=90] at (" << j + 0.5 << ", " << I + 0.2 << ") { "
                 << sd.convert(source[j]) << " };\n";
        for (unsigned i = 0; i < I; ++i)
            cout << "\\node[anchor=west] at (" << J + 0.2 << ", " << I - i - 0.5 << ") { " << td.convert(target[i])
                 << " };\n";

        float eps = 0.01;
        for (unsigned i = 0; i < I; ++i) {
            for (unsigned j = 0; j < J; ++j) {
                float v = TensorTools::access_element(a, Dim({(unsigned int) j, (unsigned int) i}));
                //int val = int(pow(v, 0.5) * 100);
                int val = int(v * 100);
                cout << "\\fill[blue!" << val << "!black] (" << j + eps << ", " << I - i - 1 + eps << ") rectangle ("
                     << j + 1 - eps << "," << I - i - eps << ");\n";
            }
        }
        cout << "\\draw[step=1cm,color=gray] (0,0) grid (" << J << ", " << I << ");\n";
        cout << "\\end{tikzpicture}\n";
    }


    template<class Builder>
    std::vector<int>
    AttentionalModel<Builder>::Greedy_Decode(const std::vector<int> &source, ComputationGraph &cg,
                                             dynet::Dict &tdict, const std::vector<int> *ctx) {
        const int sos_sym = tdict.convert("<s>");
        const int eos_sym = tdict.convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        //std::cerr << tdict.convert(target.back());
        unsigned t = 0;
        StartNewInstance(source, cg, ctx);
        while (target.back() != eos_sym) {
            Expression i_scores = AddInput(target.back(), t, cg);
            Expression ydist = softmax(i_scores); // compiler warning, but see below

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = as_vector(cg.incremental_forward(ydist));
            auto pr_w = dist[w];
            for (unsigned x = 1; x < dist.size(); ++x) {
                if (dist[x] > pr_w) {
                    w = x;
                    pr_w = dist[x];
                }
            }

            // break potential infinite loop
            if (t > 2 * source.size()) {
                w = eos_sym;
                pr_w = dist[w];
            }

            //std::cerr << " " << tdict.convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }
        //std::cerr << std::endl;

        return target;
    }

    struct Hypothesis {
        Hypothesis() {};

        Hypothesis(RNNPointer state, int tgt, float cst, std::vector <Expression> &al)
                : builder_state(state), target({tgt}), cost(cst), aligns(al) {}

        Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last, std::vector <Expression> &al)
                : builder_state(state), target(last.target), cost(cst), aligns(al) {
            target.push_back(tgt);
        }

        RNNPointer builder_state;
        std::vector<int> target;
        float cost;
        std::vector <Expression> aligns;
    };

    template<class Builder>
    std::vector<int>
    AttentionalModel<Builder>::Beam_Decode(const std::vector<int> &source, ComputationGraph &cg,
                                           unsigned beam_width, dynet::Dict &tdict, const std::vector<int> *ctx) {
        const unsigned sos_sym = tdict.convert("<s>");
        const unsigned eos_sym = tdict.convert("</s>");

        StartNewInstance(source, cg, ctx);

        std::vector <Hypothesis> chart;
        chart.push_back(Hypothesis(builder.state(), sos_sym, 0.0f, aligns));

        std::vector<unsigned> vocab(boost::copy_range < std::vector < unsigned >> (boost::irange(0u, tgt_vocab_size)));
        std::vector <Hypothesis> completed;

        for (unsigned steps = 0; completed.size() < beam_width && steps < 2 * source.size(); ++steps) {
            std::vector <Hypothesis> new_chart;

            for (auto &hprev: chart) {
                //std::cerr << "hypo t[-1]=" << tdict.convert(hprev.target.back()) << " cost " << hprev.cost << std::endl;
                if (giza_markov || giza_fertility)
                    aligns = hprev.aligns;
                Expression i_scores = AddInput(hprev.target.back(), hprev.target.size() - 1, cg, &hprev.builder_state);
                Expression ydist = softmax(i_scores);

                // find the top k best next words
                auto dist = as_vector(cg.incremental_forward(ydist));
                std::partial_sort(vocab.begin(), vocab.begin() + beam_width, vocab.end(),
                                  [&dist](unsigned v1, unsigned v2) { return dist[v1] > dist[v2]; });

                // add to chart
                for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
                    //std::cerr << "\t++word " << tdict.convert(*vi) << " prob " << dist[*vi] << std::endl;
                    //if (new_chart.size() < beam_width) {
                    Hypothesis hnew(builder.state(), *vi, hprev.cost - std::log(dist[*vi]), hprev, aligns);
                    if (*vi == eos_sym)
                        completed.push_back(hnew);
                    else
                        new_chart.push_back(hnew);
                    //}
                }
            }

            if (new_chart.size() > beam_width) {
                // sort new_chart by score, to get kbest candidates
                std::partial_sort(new_chart.begin(), new_chart.begin() + beam_width, new_chart.end(),
                                  [](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
                new_chart.resize(beam_width);
            }
            chart.swap(new_chart);
        }

        // sort completed by score, adjusting for length -- not very effective, too short!
        auto best = std::min_element(completed.begin(), completed.end(),
                                     [](Hypothesis &h1, Hypothesis &h2) {
                                         return h1.cost / h1.target.size() < h2.cost / h2.target.size();
                                     });
        assert(best != completed.end());

        return best->target;
    }

    template<class Builder>
    std::vector<int>
    AttentionalModel<Builder>::Sample(const std::vector<int> &source, ComputationGraph &cg, dynet::Dict &tdict,
                                      const std::vector<int> *ctx) {
        const int sos_sym = tdict.convert("<s>");
        const int eos_sym = tdict.convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        std::cerr << tdict.convert(target.back());
        int t = 0;
        StartNewInstance(source, cg, ctx);
        while (target.back() != eos_sym) {
            Expression i_scores = AddInput(target.back(), t, cg);
            Expression ydist = softmax(i_scores);

            // in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
            auto dist = as_vector(cg.incremental_forward(ydist));
            double p = rand01();
            unsigned w = 0;
            for (; w < dist.size(); ++w) {
                p -= dist[w];
                if (p < 0) break;
            }
            // this shouldn't happen
            if (w == dist.size()) w = eos_sym;

            std::cerr << " " << tdict.convert(w) << " [p=" << dist[w] << "]";
            t += 1;
            target.push_back(w);
        }
        std::cerr << std::endl;

        return target;
    }

    template<class Builder>
    void AttentionalModel<Builder>::Display_Fertility(const std::vector<int> &source, Dict &sd) {
        ComputationGraph cg;
        StartNewInstance(source, cg);
        assert(global_fertility);

        Expression fbias = concatenate_cols(std::vector<Expression>(slen, parameter(cg, p_bfhid)));
        Expression mbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfmu)));
        Expression vbias = concatenate(std::vector<Expression>(slen, parameter(cg, p_bfvar)));
        Expression fhid = tanh(transpose(fbias + parameter(cg, p_Wfhid) * src));
        Expression mu = mbias + fhid * parameter(cg, p_Wfmu);
        auto mu_vec = as_vector(cg.incremental_forward(mu));
        Expression var = exp(vbias + fhid * parameter(cg, p_Wfvar));
        auto var_vec = as_vector(cg.incremental_forward(var));

        for (unsigned j = 1; j < slen - 1; ++j)
            std::cout << sd.convert(source[j]) << '\t' << mu_vec[j] << '\t' << var_vec[j] << '\n';
    }

    template<class Builder>
    void AttentionalModel<Builder>::Display_Empirical_Fertility(const std::vector<int> &source,
                                                                const std::vector<int> &target, Dict &sd) {
        ComputationGraph cg;
        Expression alignment;
        ModelStats stats;
        BuildGraph(source, target, cg, stats, &alignment);

        Expression totals = sum_cols(alignment);
        auto totals_vec = as_vector(cg.incremental_forward(totals));

        for (unsigned j = 0; j < slen; ++j)
            std::cout << sd.convert(source[j]) << '\t' << totals_vec[j] << '\n';
    }

#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace dynet
