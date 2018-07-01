#include "attentional.h" // AM
//#include "rnnlm.h" // RNNLM
#include "ensemble-decoder.h"

#include "dict-utils.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>

//================
using namespace std;
using namespace dynet;
using namespace boost::program_options;

//dynet::Dict sdict; //global source vocab
//dynet::Dict tdict; //global target vocab

//=================
typedef tuple <Sentence, Sentence> SentencePair;
typedef vector <SentencePair> ParaCorpus;
typedef vector <Sentence> MonoCorpus;

//================
int main_body(variables_map vm);

ParaCorpus Read_ParaCorpus(const string &filename, unsigned slen, dynet::Dict &sdict, dynet::Dict &tdict);

MonoCorpus Read_Corpus(const string &filename, unsigned slen, dynet::Dict &sdict, dynet::Dict &tdict);

void initialise_model(Model &model, const string &filename);

template<class AM_t>
void Test_Decode(Model &model, AM_t *pam, const ParaCorpus &test_cor, dynet::Dict &td, unsigned beam, string out_fname);

template <class AM_t>
void Test_pplx(Model &model, vector<AM_t *> seq2seq, bool enc_shared, bool dec_shared, double epoch_frac,
               vector<float> eta, vector<float> gamma,
               const vector<ParaCorpus> &devel, string out_fname);

template<class AM_t>
void Multi_Task_Learn_minibatch(Model &model, vector<AM_t *> seq2seq, bool enc_shared, bool dec_shared, int num_tasks,
                                unsigned schedule,
                                unsigned minibatch_size, const vector <ParaCorpus> &train_cor,
                                const vector <ParaCorpus> &dev_cor, unsigned opt_type,
                                int lr_epochs, vector<float> eta_decay, float g_clip_threshold,
                                const vector<float> gamma, vector<float> eta, const unsigned treport_every,
                                const unsigned dreport_every,
                                const unsigned max_epoch, const string &modfile, bool adaptation);


vector <string> sep_string(string str) {
    vector <string> res;
    replace(str.begin(), str.end(), ',', ' ');
    istringstream iss(str);
    for (string s; iss >> s;) { res.push_back(s); }
    //cerr << endl;
    return res;
}

vector<float> s2f(vector <string> str) {
    vector<float> res;
    for (auto s:str) res.push_back(stof(s));
    return res;
}

vector<unsigned> s2u(vector <string> str) {
    vector<unsigned> res;
    for (auto s:str) res.push_back((unsigned) stoi(s));
    return res;
}

template<class rnn_t>
int main_body(variables_map vm);

//=================
int main(int argc, char **argv) {
    dynet::initialize(argc, argv);
    // command line processing
    variables_map vm;
    options_description opts("Allowed options");
    opts.add_options()
            ("help", "print help message")
            //------------
            ("pplx", "compute the pplx on test set")
            ("adapt", "Load a saved model and train it for ${adapt-epochs} more epochs")
            ("adapt_epochs", value<unsigned>()->default_value(20), "number of epochs for adaptation")
            //-----------------------------------------
            ("train,t", value<string>(),
             "LIST of fileS, separated by comma. Each file containing training parallel sentences for a task (for source/targer vocabulary building on-the-fly), with each line consisting of source ||| target.")
            ("dev,d", value<string>(),
             "LIST of fileS containing development parallel sentences, with each line consisting of source ||| target.")
            ("test,T", value<string>(),
             "LIST of fileS containing test parallel sentences, with each line consisting of source ||| target. In case want to skip the test for a task, specify its name as _")
            //-----------------------------------------
            ("initialise", value<string>(), "load pre-trained model parameters from file")
            ("parameters,p", value<string>()->default_value("modfile.params"), "save best parameters to this file")
            ("beam", value<unsigned>()->default_value(1), "size of beam in decoding; 1: greedy")
            //-----------------------------------------
            ("minibatch_size", value<unsigned>()->default_value(1),
             "impose the minibatch size for training (support both GPU and CPU); no by default")
            //("dev_minibatch_size", value<unsigned>()->default_value(1), "impose the minibatch size for development (support both GPU and CPU); no by default")
            ("dynet-autobatch", value<unsigned>()->default_value(0),
             "impose the auto-batch mode (support both GPU and CPU); no by default") //--dynet-autobatch 1
            //-----------------------------------------
            ("sgd_trainer", value<unsigned>()->default_value(0),
             "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)")
            //-----------------------------------------TODO => DEBUG
            ("slen_limit", value<unsigned>()->default_value(0),
             "limit the sentence length (either source or target); no by default")
            //("multitask", value<unsigned>()->default_value(2), "0: sharing encoder, 1: sharing decoder, 2: sharing encoder AND decoder  (default)")
            ("mtl_shared_layers",
             "switches between MTL layers and the MTL whole (the old version which is the default)")
            ("schedule", value<unsigned>()->default_value(0),
             "0: all tasks-order randomised (dafult), 1: all tasks-main task first others randomised ordder 2: main task, a random task, order randmised, 3: main task, a random task, main task first")
            ("enc_shared", "share the encoder")
            ("dec_shared", "share the decoder")
            ("svocab_shared", "share the source vocab")
            ("tvocab_shared", "share the target vocab")
            ("align_shared", "share the alignment component in decoders (only supported in multi-layer MTL currently)")
            //-----------------------------------------
            ("eta", value<string>(), "the eta hyper-parameter for stochastic gradient update in tuning AM modelS")
            ("lr_eta_decay", value<string>(), "SGD learning rate decay value, comma separated.")
            ("lr_epochs", value<int>()->default_value(0),
             "no. of epochs for starting learning rate annealing (e.g., halving)")
            ("g_clip_threshold", value<float>()->default_value(5.f),
             "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
            //----------------------------------------- //TODO:
            ("dropout_enc", value<float>()->default_value(0.f),
             "use dropout (Gal et al., 2016) for RNN encoder(s), e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
            ("dropout_dec", value<float>()->default_value(0.f),
             "use dropout (Gal et al., 2016) for RNN decoder, e.g., 0.5 (input=0.5;hidden=0.5;cell=0.5) for LSTM; none by default")
            //----------------------------------------- TODO
            ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
            ("lstm", "use Long Short Term Memory (LSTM) for recurrent structure; default RNN")
            ("vlstm", "use Vanilla Long Short Term Memory (VLSTM) for recurrent structure; default RNN")
            ("dglstm",
             "use Depth-Gated Long Short Term Memory (DGLSTM) (Kaisheng et al., 2015; https://arxiv.org/abs/1508.03790) for recurrent structure; default RNN")
            //-----------------------------------------
            ("gamma", value<string>(), "the gamma  hyper-parameter for stochastic gradient update in tuning AM modelS")
            ("tlayers", value<string>(), "use <num> layers for target RNN components, comma separated")
            ("slayers", value<string>(), "use <num> layers for source RNN components, comma separated")
            ("shared_tlayers", value<string>(), "use <num> SHARED layers for target RNN components, comma separated")
            ("shared_slayers", value<string>(), "use <num> SHARED layers for source RNN components, comma separated")

            //FIX ME: need to separate hidden dimention size of the decoders from the encoder: consequence is model parameters should be modified!
            //currently all of the decoders use the same hidden dimension size which is equal to the encoder
            ("hidden,h", value<unsigned>()->default_value(100), "use <num> dimensions for recurrent hidden states")
            ("align,a", value<unsigned>()->default_value(50), "use <num> dimensions for alignment projection")
            //-----------------------------------------
            ("epoch,e", value<unsigned>()->default_value(10), "number of training epochs, 10 by default")
            //-----------------------------------------
            ("treport", value<unsigned>()->default_value(100),
             "no. of training instances for reporting current model status on training data")
            ("dreport", value<unsigned>()->default_value(50000),
             "no. of training instances for reporting current model status on development data (dreport = N * treport)")
            ("verbose,v", "be extremely chatty");
    store(parse_command_line(argc, argv, opts), vm);
    notify(vm);

    if (vm.count("help") || vm.count("train") != 1 || (vm.count("dev") != 1)) {// FIXME: check the missing ones?
        cout << opts << "\n";
        return EXIT_SUCCESS;
    }

    if (vm.count("lstm"))
        return main_body<LSTMBuilder>(vm);
    else if (vm.count("vlstm"))
        return main_body<VanillaLSTMBuilder>(vm);
    else if (vm.count("dglstm"))
        return main_body<DGLSTMBuilder>(vm);
    else if (vm.count("gru"))
        return main_body<GRUBuilder>(vm);
    else
        return main_body<SimpleRNNBuilder>(vm);
}

void read_corpora(vector <ParaCorpus> &train_cor, vector <ParaCorpus> &dev_cor, vector <ParaCorpus> &test_cor,
                  vector <dynet::Dict> &sdict_vec, vector <dynet::Dict> &tdict_vec,
                  unsigned int &num_tasks, bool svocab_shared, bool tvocab_shared, bool enc_shared, bool dec_shared,
                  variables_map vm) {

    vector <string> train_files = sep_string(vm["train"].as<string>());
    unsigned slen = vm["slen_limit"].as<unsigned>();

    num_tasks = train_files.size();

    cerr << "Reading training parallel data from " << vm["train"].as<string>() << "...\n";
    for (auto k = 0; k < num_tasks; k++) {
        dynet::Dict temp_tgt_dict;
        tdict_vec.push_back(temp_tgt_dict);
        kTGT_SOS = tdict_vec[k].convert("<s>");
        kTGT_EOS = tdict_vec[k].convert("</s>");
        kTGT_UNK = tdict_vec[k].convert("<unk>");

        dynet::Dict temp_src_dict;
        sdict_vec.push_back(temp_src_dict);
        kTGT_SOS = sdict_vec[k].convert("<s>");
        kTGT_EOS = sdict_vec[k].convert("</s>");
        kTGT_UNK = sdict_vec[k].convert("<unk>");

        unsigned tvocab_index = ((tvocab_shared) || (dec_shared)) ? 0 : k;
        unsigned svocab_index = ((svocab_shared) || (enc_shared)) ? 0 : k;
        train_cor.push_back(Read_ParaCorpus(train_files[k], slen, sdict_vec[svocab_index], tdict_vec[tvocab_index]));
    }

    if (vm.count("dev")) {
        vector <string> dev_files = sep_string(vm["dev"].as<string>());
        cerr << "Reading dev parallel data from " << vm["dev"].as<string>() << "...\n";
        for (auto k = 0; k < num_tasks; k++) {
            tdict_vec[k].freeze();  // no new word types allowed
            sdict_vec[k].freeze();  // no new word types allowed

            unsigned tvocab_index = ((tvocab_shared) || (dec_shared)) ? 0 : k;
            unsigned svocab_index = ((svocab_shared) || (enc_shared)) ? 0 : k;
            dev_cor.push_back(Read_ParaCorpus(dev_files[k], slen, sdict_vec[svocab_index], tdict_vec[tvocab_index]));
        }
    }

    //vector<MonoCorpus> test_cor;
    if (vm.count("test")) {
        vector <string> test_files = sep_string(vm["test"].as<string>());
        cerr << "Reading test examples from " << vm["test"].as<string>() << endl;
        //testing = Read_Corpus(vm["test"].as<string>(), slen, sdict_vec[0], tdict_vec[0]);
        for (auto k = 0; k < num_tasks; k++) {
            tdict_vec[k].freeze();  // no new word types allowed
            sdict_vec[k].freeze();  // no new word types allowed


            if (test_files[k] != "_") {
                unsigned tvocab_index = ((tvocab_shared) || (dec_shared)) ? 0 : k;
                unsigned svocab_index = ((svocab_shared) || (enc_shared)) ? 0 : k;
                test_cor.push_back(
                        Read_ParaCorpus(test_files[k], slen, sdict_vec[svocab_index], tdict_vec[tvocab_index]));
            }
        }
    }

}


template<class rnn_t>
void build_whole_shared_mtl(bool enc_shared, bool dec_shared, bool tvocab_shared, bool svocab_shared,
                            vector <dynet::Dict> &sdict_vec, vector <dynet::Dict> &tdict_vec,
                            unsigned hidden_dim, unsigned align_dim, unsigned num_tasks, vector<unsigned> slayers,
                            vector<unsigned> tlayers,
                            float enc_drop, float dec_drop, vector<SharedDecoder<rnn_t> *> &pdec_vec,
                            vector<SharedEncoder<rnn_t> *> &penc_vec,
                            vector<AttentionalModel<rnn_t> *> &seq2seq, Model &model) {
    if (enc_shared && dec_shared) { //shared both
        SharedEncoder<rnn_t> *shared_penc = new SharedEncoder<rnn_t>(&model, sdict_vec[0].size(), slayers[0],
                                                                     hidden_dim, nullptr);
        penc_vec.push_back(shared_penc);
        SharedDecoder<rnn_t> *shared_pdec = new SharedDecoder<rnn_t>(&model, tdict_vec[0].size(), tlayers[0],
                                                                     hidden_dim, align_dim, true,
                                                                     false, false, false, false, false, nullptr);
        pdec_vec.push_back(shared_pdec);
        //-----
        AttentionalModel<rnn_t> *p_temp = new AttentionalModel<rnn_t>(&model, shared_penc, shared_pdec,
                                                                      sdict_vec[0].size(), tdict_vec[0].size(),
                                                                      hidden_dim, align_dim,
                                                                      true, false, false, false, false, false);
        p_temp->Set_Dropout(enc_drop, dec_drop);
        seq2seq.push_back(p_temp);

    } else {
        if (enc_shared) { //shared encoder
            SharedEncoder<rnn_t> *shared_penc = new SharedEncoder<rnn_t>(&model, sdict_vec[0].size(), slayers[0],
                                                                         hidden_dim, nullptr);
            penc_vec.push_back(shared_penc);
            for (auto k = 0; k < num_tasks; k++) {
                SharedDecoder<rnn_t> *pdec;
                if (!tvocab_shared)
                    pdec = new SharedDecoder<rnn_t>(&model, tdict_vec[k].size(), tlayers[k], hidden_dim, align_dim,
                                                    true,
                                                    false, false, false, false, false, nullptr);
                else {
                    pdec = new SharedDecoder<rnn_t>(&model, tdict_vec[0].size(), tlayers[k], hidden_dim, align_dim,
                                                    true,
                                                    false, false, false, false, false,
                                                    (k == 0) ? (LookupParameter * ) nullptr : &(pdec_vec[0]->p_ct));
                }
                pdec_vec.push_back(pdec);

                AttentionalModel<rnn_t> *p_temp = new AttentionalModel<rnn_t>(&model, shared_penc, pdec,
                                                                              sdict_vec[0].size(),
                                                                              tdict_vec[k].size(), hidden_dim,
                                                                              align_dim, true, false, false, false,
                                                                              false, false);
                p_temp->Set_Dropout(enc_drop, dec_drop);
                seq2seq.push_back(p_temp);
            }
        } else {
            if (dec_shared) { //shared decoder
                SharedDecoder<rnn_t> *shared_pdec = new SharedDecoder<rnn_t>(&model, tdict_vec[0].size(), tlayers[0],
                                                                             hidden_dim, align_dim, true,
                                                                             false, false, false, false, false,
                                                                             nullptr);
                pdec_vec.push_back(shared_pdec);
                //cerr << "aa33 " << num_tasks << endl;
                for (auto k = 0; k < num_tasks; k++) {
                    //cerr << "aa33-00" << endl;
                    SharedEncoder<rnn_t> *penc;
                    if (!svocab_shared)
                        penc = new SharedEncoder<rnn_t>(&model, sdict_vec[k].size(), slayers[k], hidden_dim, nullptr);
                    else
                        penc = new SharedEncoder<rnn_t>(&model, sdict_vec[0].size(), slayers[k], hidden_dim,
                                                        (k == 0) ? (LookupParameter * ) nullptr: &(penc_vec[0]->p_cs));
                    //cerr << "aa33-11" << endl;
                    penc_vec.push_back(penc);
                    //cerr << "aa33-22" << endl;
                    AttentionalModel<rnn_t> *p_temp = new AttentionalModel<rnn_t>(&model, penc, shared_pdec,
                                                                                  sdict_vec[k].size(),
                                                                                  tdict_vec[0].size(), hidden_dim,
                                                                                  align_dim, true, false, false, false,
                                                                                  false, false);
                    p_temp->Set_Dropout(enc_drop, dec_drop);
                    //cerr << "aa33-33" << endl;
                    seq2seq.push_back(p_temp);
                    //cerr << "aa33-44" << endl;
                }
                //cerr << "aa44" << endl;
            } else {
                cerr << "multi-task configuration is not known, sorry!";
                abort();
            }
        }
    }
}

template<class rnn_t>
void build_layers_shared_mtl(bool align_shared, bool tvocab_shared, bool svocab_shared,
                             vector <dynet::Dict> &sdict_vec, vector <dynet::Dict> &tdict_vec,
                             unsigned hidden_dim, unsigned align_dim, unsigned num_tasks,
                             vector<unsigned> slayers, vector<unsigned> tlayers, vector<unsigned> shared_slayers,
                             vector<unsigned> shared_tlayers,
                             float enc_drop, float dec_drop, vector<SharedDecoder<rnn_t> *> &pdec_vec,
                             vector<SharedEncoder<rnn_t> *> &penc_vec,
                             vector<AttentionalModel<rnn_t> *> &seq2seq, Model &model) {

    SharedEncoder<rnn_t> *main_penc = new SharedEncoder<rnn_t>(&model, sdict_vec[0].size(), slayers[0], hidden_dim,
                                                               nullptr);
    SharedDecoder<rnn_t> *main_pdec = new SharedDecoder<rnn_t>(&model, tdict_vec[0].size(), tlayers[0], hidden_dim,
                                                               align_dim, true,
                                                               false, false, false, false, false, nullptr);
    penc_vec.push_back(main_penc);
    pdec_vec.push_back(main_pdec);

    AttentionalModel<rnn_t> *p_main = new AttentionalModel<rnn_t>(&model, main_penc, main_pdec, sdict_vec[0].size(),
                                                                  tdict_vec[0].size(),
                                                                  hidden_dim, align_dim, true, false, false, false,
                                                                  false, false);
    cout << "dic sizes in model 0 are " << tdict_vec[0].size() << " " << sdict_vec[0].size() << endl;
    p_main->Set_Dropout(enc_drop, dec_drop);
    seq2seq.push_back(p_main);

    for (auto k = 1; k < num_tasks; k++) {
        auto tvsize = (tvocab_shared) ? tdict_vec[0].size() : tdict_vec[k].size();
        auto svsize = (svocab_shared) ? sdict_vec[0].size() : sdict_vec[k].size();
        cout << "dic sizes in model " << k << " are " << tvsize << " " << svsize << endl;

        SharedDecoder<rnn_t> *shared_pdec = new SharedDecoder<rnn_t>(&model, tvsize,
                                                                     tlayers[0], shared_tlayers[k], hidden_dim,
                                                                     align_dim, true, false, false, false, false, false,
                                                                     tvocab_shared, align_shared, main_pdec);
        pdec_vec.push_back(shared_pdec);
        //---
        SharedEncoder<rnn_t> *shared_penc = new SharedEncoder<rnn_t>(&model, svsize,
                                                                     slayers[0], shared_slayers[k], hidden_dim,
                                                                     svocab_shared, main_penc);
        penc_vec.push_back(shared_penc);
        //-----
        AttentionalModel<rnn_t> *p_temp = new AttentionalModel<rnn_t>(&model, shared_penc, shared_pdec, svsize, tvsize,
                                                                      hidden_dim, align_dim, true, false, false, false,
                                                                      false, false);
        p_temp->Set_Dropout(enc_drop, dec_drop);
        seq2seq.push_back(p_temp);
    }

}

template<class rnn_t>
int main_body(variables_map vm) {

    //--------------------------------------------------------------
    // read the commandline switches--------------------------------

    unsigned treport = vm["treport"].as<unsigned>();
    unsigned dreport = vm["dreport"].as<unsigned>();

    vector<float> gamma = s2f(sep_string(vm["gamma"].as<string>()));
    //unsigned multitask = vm["multitask"].as<unsigned>();
    bool svocab_shared = false;
    bool tvocab_shared = false;
    bool align_shared = false;
    bool enc_shared = false;
    bool dec_shared = false;
    bool adaptation = false;
    unsigned schedule = vm["schedule"].as<unsigned>();
    unsigned num_tasks = 0;

    vector<unsigned> slayers = s2u(sep_string(vm["slayers"].as<string>()));
    vector<unsigned> tlayers = s2u(sep_string(vm["tlayers"].as<string>()));
    vector<unsigned> shared_slayers, shared_tlayers; // used in the 2nd MTL architecture
    unsigned hidden_dim = vm["hidden"].as<unsigned>();
    unsigned align_dim = vm["align"].as<unsigned>();
    float enc_drop = vm["dropout_enc"].as<float>();
    float dec_drop = vm["dropout_dec"].as<float>();

    unsigned adapt_epochs = vm["adapt_epochs"].as<unsigned>();
    unsigned max_epoch = vm["epoch"].as<unsigned>();
    unsigned opt_type = vm["sgd_trainer"].as<unsigned>();
    unsigned minibatch_szie = vm["minibatch_size"].as<unsigned>();
    vector<float> eta = s2f(sep_string(vm["eta"].as<string>()));
    float g_clip_threshold = vm["g_clip_threshold"].as<float>();
    vector<float> lr_eta_decay = s2f(sep_string(vm["lr_eta_decay"].as<string>()));
    float lr_epochs = vm["lr_epochs"].as<int>();

    bool shared_layers_mtl = false;
    cerr<<"schedule: "<< to_string(schedule)<<endl;
    if (vm.count("mtl_shared_layers")) {
        shared_layers_mtl = true;
        cerr << "mtl_shared_layers is on" << endl;
        shared_tlayers = s2u(sep_string(vm["shared_tlayers"].as<string>()));
        shared_slayers = s2u(sep_string(vm["shared_slayers"].as<string>()));
    } else
        cerr << "shared_whole_mtl is off" << endl;

    if (vm.count("align_shared")) {
        align_shared = true;
        cerr << "align_shared is on" << endl;
    }
    if (vm.count("svocab_shared")) {
        svocab_shared = true;
        cerr << "svocab_shared is on" << endl;
    }
    if (vm.count("tvocab_shared")) {
        tvocab_shared = true;
        cerr << "tvocab_shared is on" << endl;
    }
    if (vm.count("enc_shared")) {
        enc_shared = true;
        cerr << "enc_shared is on" << endl;
    }
    if (vm.count("dec_shared")) {
        dec_shared = true;
        cerr << "dec_shared is on" << endl;
    }
    if (vm.count("adapt")) {
        adaptation = true;
        max_epoch = adapt_epochs;
        cerr << "adaptation is on" << endl;
    }

    string flavour = "RNN";
    if (vm.count("lstm"))
        flavour = "LSTM";
    else if (vm.count("vlstm"))
        flavour = "VanillaLSTM";
    else if (vm.count("dglstm"))
        flavour = "DGLSTM";
    else if (vm.count("gru"))
        flavour = "GRU";

    string modfile;
    if (vm.count("parameters"))
        modfile = vm["parameters"].as<string>();
    else {
        ostringstream os;
        os << "multi_task_1encoder_Ndecoder"
           << '_' << vm["slayers"].as<string>()
           << '_' << vm["tlayers"].as<string>()
           << '_' << hidden_dim
           << '_' << align_dim
           << '_' << flavour
           << "-pid" << getpid() << ".params";
        modfile = os.str();
    }
    cerr << "Parameters will be written to: " << modfile << endl;

    //--------------------------------------------------------------
    // read the corpora and create vocabs --------------------------
    vector <ParaCorpus> train_cor, dev_cor, test_cor;
    vector <dynet::Dict> tdict_vec; //global target vocab
    vector <dynet::Dict> sdict_vec; //global source vocab

    read_corpora(train_cor, dev_cor, test_cor, sdict_vec, tdict_vec,
                 num_tasks, svocab_shared, tvocab_shared, enc_shared, dec_shared, vm);

    //--------------------------------------------------------------
    // --- Build the MTL architectures -----------------------------
    //cerr << "aa11" << endl;
    Model model;
    vector < AttentionalModel<rnn_t> * > seq2seq;
    //cerr << "aa22" << endl;
    vector < SharedDecoder<rnn_t> * > pdec_vec;
    vector < SharedEncoder<rnn_t> * > penc_vec;
    if (!shared_layers_mtl) {
        //cout << "im here AA 11\n";
        build_whole_shared_mtl<rnn_t>(enc_shared, dec_shared, tvocab_shared, svocab_shared, sdict_vec, tdict_vec,
                                      hidden_dim, align_dim, num_tasks, slayers, tlayers, enc_drop, dec_drop, pdec_vec,
                                      penc_vec, seq2seq, model);
    } else
        build_layers_shared_mtl<rnn_t>(align_shared, tvocab_shared, svocab_shared, sdict_vec, tdict_vec,
                                       hidden_dim, align_dim, num_tasks, slayers, tlayers, shared_slayers,
                                       shared_tlayers,
                                       enc_drop, dec_drop, pdec_vec, penc_vec, seq2seq, model);


    //cerr << "aa44 " << num_tasks << endl;

    //--------------------------------------------------------------
    // doing the task: training or testing -------------------------
    if (vm.count("initialise")){
        initialise_model(model, vm["initialise"].as<string>());
    } else{
        if (adaptation){
            cerr << "Cannot run adaptation without loading pre-trained model";
            return EXIT_SUCCESS;
        }
    }
    //cerr << "aa55" << endl;
    // call training functions
    if (vm.count("test") > 0) { //TODO ZZZZZZ
        cerr << "Decoding..." << endl;
        vector<string> test_files = sep_string(vm["test"].as<string>());
        if (vm.count("pplx")){
            cerr << "Computing pplx..." << endl;
            string pplx_fname = vm["initialise"].as<string>() + ".pplx";
            Test_pplx(model, seq2seq, enc_shared, dec_shared, 50, eta, gamma, test_cor, pplx_fname);
        }else {
            // info:
            for (auto k = 0; k < num_tasks; k++) {
                //for (auto k = 0; k < 1; k++) {
                string test_fname = test_files[k];
                if (test_fname == "_") continue;
                string out_fname = vm["initialise"].as<string>() + ".out.t" + std::to_string(k) + ".encoded";
                unsigned model_index = (enc_shared && dec_shared) ? 0 : k;
                unsigned tdic_index = (dec_shared || tvocab_shared) ? 0 : k;
                Test_Decode(model, seq2seq[model_index], test_cor[k], tdict_vec[tdic_index], vm["beam"].as<unsigned>(),
                            out_fname);
            }
        }
    } else {
        cerr << "Training...with mini-batch size " << minibatch_szie << endl;
        Multi_Task_Learn_minibatch<AttentionalModel<rnn_t> >(model, seq2seq, enc_shared, dec_shared, num_tasks,
                                                             schedule, minibatch_szie,
                                                             train_cor, dev_cor, opt_type, lr_epochs, lr_eta_decay,
                                                             g_clip_threshold,
                                                             gamma, eta, treport, dreport, max_epoch, modfile, adaptation);
    }

    //--------------------------------------------------------------
    // cleaning up ... ---------------------------------------------
    for (auto k = 0; k < pdec_vec.size(); k++) delete pdec_vec[k];
    for (auto k = 0; k < penc_vec.size(); k++) delete penc_vec[k];

    if (vm.count("test") == 0)
        for (auto k = 0; k < seq2seq.size(); k++) delete seq2seq[k];
    //cout << "666" << endl;

    return EXIT_SUCCESS;
}

//==================
void initialise_model(Model &model, const string &filename) {
    cerr << "Initialising model parameters from file: " << filename << endl;
    //ifstream in(filename, ifstream::in);
    //boost::archive::text_iarchive ia(in);
    //ia >> model;
    dynet::load_dynet_model(filename, &model);// FIXME: use binary streaming instead for saving disk spaces
}

//==================
template <class AM_t>
//void Test_Decode(Model &model, AM_t &am, string test_file, bool doco, unsigned beam, bool r2l_target)
void Test_Decode(Model &model, AM_t  * pam, const ParaCorpus &test_cor,   dynet::Dict &td, unsigned beam, string out_fname)
{
    int lno = 0;

    std::ofstream ofs (out_fname, std::ofstream::out);
    //EnsembleDecoder<AM_t> edec(std::vector<AM_t*>({&am}), &td);//FIXME: single decoder only
    std::shared_ptr<AM_t> sh_pam(nullptr);
    sh_pam.reset(pam);
    std::vector<std::shared_ptr<AM_t> > v_ams;
    v_ams.push_back(sh_pam);
    EnsembleDecoder<AM_t> edec(v_ams, &td);//FIXME: multiple decoders
    edec.SetBeamSize(beam);

    Timer timer_dec("completed in");

    for (auto sent_pair : test_cor) {

        Sentence source, true_target, gen_target;
        tie(source,true_target) = sent_pair;

        ComputationGraph cg;

        if (beam > 0) {
            // Trevor's beam search implementation
            //target = am.Beam_Decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);// ensemble decoding not supported yet!

            // Vu's beam search implementation
            EnsembleDecoderHypPtr trg_hyp = edec.Generate(source, cg);//1-best
            if (trg_hyp.get() == nullptr) {
                gen_target.clear();
                //align.clear();
            } else {
                gen_target = trg_hyp->GetSentence();
                //align = trg_hyp->GetAlignment();
                //str_trg = Convert2iStr(*vocab_trg, sent_trg, false);
                //MapWords(str_src, sent_trg, align, mapping, str_trg);
            }
        }
        bool first = true;
        for (auto &w: gen_target) {
            if (!first) ofs << " ";
            ofs << td.convert(w);
            first = false;
        }
        ofs << endl;
        lno++;
    }

    double elapsed = timer_dec.elapsed();
    cerr << "Decoding is finished!" << endl;
    cerr << "The results are written into " << out_fname << endl;
    cerr << "Decoded " << lno << " sentences, completed in " << elapsed/1000 << "(s)" << endl;
    ofs.close();
}

//=============================================

template <class AM_t>
//void Test_pplx(Model &model, vector<AM_t *> seq2seq , const ParaCorpus &test_cor,   dynet::Dict &td, unsigned beam, string out_fname)
void Test_pplx(Model &model, vector<AM_t *> seq2seq, bool enc_shared, bool dec_shared, double epoch_frac,
               vector<float> eta, vector<float> gamma,
               const vector<ParaCorpus> &devel, string out_fname)
{
    std::ofstream ofs (out_fname, std::ofstream::out);
    Sentence ssent, tsent;
    double total_loss = 0;
    double avg_pplx = 0;
    double total_length = 0;
    int total_sent = 0;
    Timer timer_iteration("completed in");
    vector<double> res_loss;

    float total_gamma = 0;
    for (auto g : gamma) total_gamma += g;

    cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    for (auto j = 0; j < devel.size(); j++) {
        ModelStats dstats;

        unsigned model_index = (enc_shared && dec_shared)?0:j;
        seq2seq[model_index]->Disable_Dropout();
        timer_iteration.reset();
        for (unsigned i = 0; i < devel[j].size(); ++i) {
            tie(ssent, tsent) = devel[j][i];
            ComputationGraph cg;
            //cerr<< "here1"<<endl;
            //auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr, nullptr);
            auto  i_xent = seq2seq[model_index]->BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr, nullptr);
            //cerr<< "here2"<<endl;
            dstats.loss += as_scalar(cg.forward(i_xent));
            //cerr<< "here3"<<endl;
            //dstats.words_tgt += temp_stats.words_tgt;
            //dstats.words_src_unk += temp_stats.words_src_unk;
            //dstats.words_tgt_unk += temp_stats.words_tgt_unk;
            //cerr << "-->" << temp_stats.words_tgt << endl;
        }
        seq2seq[model_index]->Enable_Dropout();// enable dropout

        //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        ofs << "***DEV_" << j << " [epoch=" << epoch_frac << " eta=" << eta[j] << "]" << " sents=" << devel[j].size()
            << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E="
            << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ' << endl;
        timer_iteration.show();
        //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        // cerr << "-->" << dstats.words_tgt << endl;
        // some overal stats
        total_loss +=  gamma[j]* dstats.loss/total_gamma;
        total_length += gamma[j] * dstats.words_tgt/total_gamma;
        avg_pplx += gamma[j] * exp(dstats.loss / dstats.words_tgt)/total_gamma;
        total_sent += devel[j].size();
        res_loss.push_back(dstats.loss);
    }

    //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    ofs << "***DEV_TOTAL [epoch=" << epoch_frac  << "]" << " sents=" << total_sent << " total_length=" << total_length
        <<   " total_loss=" << total_loss/total_length << " total_ppl=" << exp(total_loss / total_length)
        <<   " avg_ppl=" << avg_pplx << endl;
    //timer_iteration.show();
    cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    return;
}

//=============================================
template<class AM_t>
vector<double>
report_dev_eval(Model &model, vector<AM_t *> seq2seq, bool enc_shared, bool dec_shared, double epoch_frac,
                vector<float> eta, vector<float> gamma,
                const vector <ParaCorpus> &devel, double &best_loss, string out_file) {

    Sentence ssent, tsent;
    double total_loss = 0;
    double avg_pplx = 0;
    double total_length = 0;
    int total_sent = 0;
    Timer timer_iteration("completed in");
    vector<double> res_loss;

    float total_gamma = 0;
    for (auto g : gamma) total_gamma += g;

    cerr << "--------------------------------------------------------------------------------------------------------"
         << endl;
    for (auto j = 0; j < devel.size(); j++) {
        ModelStats dstats;

        unsigned model_index = (enc_shared && dec_shared) ? 0 : j;
        seq2seq[model_index]->Disable_Dropout();
        timer_iteration.reset();
        for (unsigned i = 0; i < devel[j].size(); ++i) {
            tie(ssent, tsent) = devel[j][i];
            ComputationGraph cg;
            //auto i_xent = am.BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr, nullptr);
            auto i_xent = seq2seq[model_index]->BuildGraph(ssent, tsent, cg, dstats, nullptr, nullptr, nullptr,
                                                           nullptr);

            dstats.loss += as_scalar(cg.forward(i_xent));
            //dstats.words_tgt += temp_stats.words_tgt;
            //dstats.words_src_unk += temp_stats.words_src_unk;
            //dstats.words_tgt_unk += temp_stats.words_tgt_unk;
            //cerr << "-->" << temp_stats.words_tgt << endl;
        }
        seq2seq[model_index]->Enable_Dropout();// enable dropout

        if ((dstats.loss < best_loss) && (j == 0)) { //j == 0 must be the MT task
            best_loss = dstats.loss;
            //ofstream out(out_file, ofstream::out);
            //boost::archive::text_oarchive oa(out);
            //oa << model;
            dynet::save_dynet_model(out_file, &model);// FIXME: use binary streaming instead for saving disk spaces
        }
        //else{
        //	sgd.eta *= 0.5;
        //}
        //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        cerr << "***DEV_" << j << " [epoch=" << epoch_frac << " eta=" << eta[j] << "]" << " sents=" << devel[j].size()
             << " src_unks=" << dstats.words_src_unk << " trg_unks=" << dstats.words_tgt_unk << " E="
             << (dstats.loss / dstats.words_tgt) << " ppl=" << exp(dstats.loss / dstats.words_tgt) << ' ';
        timer_iteration.show();
        //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
        // cerr << "-->" << dstats.words_tgt << endl;
        // some overal stats
        total_loss += gamma[j] * dstats.loss / total_gamma;
        total_length += gamma[j] * dstats.words_tgt / total_gamma;
        avg_pplx += gamma[j] * exp(dstats.loss / dstats.words_tgt) / total_gamma;
        total_sent += devel[j].size();
        res_loss.push_back(dstats.loss);
    }

    //cerr << "--------------------------------------------------------------------------------------------------------" << endl;
    cerr << "***DEV_TOTAL [epoch=" << epoch_frac << "]" << " sents=" << total_sent << " total_length=" << total_length
         << " total_loss=" << total_loss / total_length << " total_ppl=" << exp(total_loss / total_length)
         << " avg_ppl=" << avg_pplx << endl;
    //timer_iteration.show();
    cerr << "--------------------------------------------------------------------------------------------------------"
         << endl;
    return res_loss;
}

//====================================
struct DoubleLength {
    DoubleLength(const ParaCorpus &cor_) : cor(cor_) {}

    bool operator()(int i1, int i2);

    const ParaCorpus &cor;
};

bool DoubleLength::operator()(int i1, int i2) {
    if (std::get<0>(cor[i2]).size() != std::get<0>(cor[i1]).size())
        return (std::get<0>(cor[i2]).size() < std::get<0>(cor[i1]).size());
    return (std::get<1>(cor[i2]).size() < std::get<1>(cor[i1]).size());
}

inline size_t Calc_Size(const Sentence &src, const Sentence &trg) {
    return src.size() + trg.size();
}

inline size_t
Create_MiniBatches(const ParaCorpus &cor, size_t max_size, std::vector <std::vector<Sentence>> &train_src_minibatch,
                   std::vector <std::vector<Sentence>> &train_trg_minibatch) {
    train_src_minibatch.clear();
    train_trg_minibatch.clear();

    std::vector <size_t> train_ids(cor.size());
    std::iota(train_ids.begin(), train_ids.end(), 0);
    if (max_size > 1)
        sort(train_ids.begin(), train_ids.end(), DoubleLength(cor));

    std::vector <Sentence> train_src_next;
    std::vector <Sentence> train_trg_next;

    size_t max_len = 0;
    for (size_t i = 0; i < train_ids.size(); i++) {
        max_len = std::max(max_len, Calc_Size(std::get<0>(cor[train_ids[i]]), std::get<1>(cor[train_ids[i]])));
        train_src_next.push_back(std::get<0>(cor[train_ids[i]]));
        train_trg_next.push_back(std::get<1>(cor[train_ids[i]]));

        if ((train_trg_next.size() + 1) * max_len > max_size) {
            train_src_minibatch.push_back(train_src_next);
            train_src_next.clear();
            train_trg_minibatch.push_back(train_trg_next);
            train_trg_next.clear();
            max_len = 0;
        }
    }

    if (train_trg_next.size()) {
        train_src_minibatch.push_back(train_src_next);
        train_trg_minibatch.push_back(train_trg_next);
    }

    // Create a sentence list for this minibatch
    //train_ids_minibatch.resize(train_src_minibatch.size());
    //std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

    return train_ids.size();
}

//==============================================
class DataIter {
public:
    DataIter(size_t N, string _name) {
        for (auto i = 0; i < N; i++) idx.push_back(i);
        name = _name;
        curr = epochs = 0;
        cerr << "data iterator for " << name << " is created; the size is " << N << endl;
    }

    ~DataIter() {}

    void idx_shuffle(void) {
        shuffle(idx.begin(), idx.end(), *rndeng);
        cerr << "*** shuffeled iterator " << name << endl;
    }

    size_t idx_next(void) {
        curr++;
        if (curr == idx.size()) {
            curr = 0;
            epochs++;
            this->idx_shuffle();
        }
        return curr;
    }

    size_t get_id() { return curr; }

    size_t get_epochs() { return epochs; }

    bool at_final(void) { return (curr == (idx.size() - 1)); }

    size_t size(void) { return idx.size(); }

private:
    vector<unsigned> idx;
    string name;
    size_t curr;
    size_t epochs;
};

template<class AM_t>
void Multi_Task_Learn_minibatch(Model &model, vector<AM_t *> seq2seq, bool enc_shared, bool dec_shared, int num_tasks,
                                unsigned schedule, unsigned minibatch_size,
                                const vector <ParaCorpus> &train_cor, const vector <ParaCorpus> &dev_cor,
                                unsigned opt_type,
                                int lr_epochs, vector<float> lr_eta_decay, float g_clip_threshold,
                                const vector<float> gamma, vector<float> eta, const unsigned treport_every,
                                const unsigned dreport_every,
                                const unsigned max_epoch, const string &modfile, bool adaptation) {
    // Create minibatches
    vector < vector < vector < Sentence > > > train_src_minibatch, train_trg_minibatch;
    vector <DataIter> train_ids_minibatch;
    //cerr << "zz11" << endl;
    for (auto i = 0; i < num_tasks; i++) {
        vector <vector<Sentence>> temp_train_src_minibatch, temp_train_trg_minibatch;
        Create_MiniBatches(train_cor[i], minibatch_size, temp_train_src_minibatch, temp_train_trg_minibatch);
        train_src_minibatch.push_back(temp_train_src_minibatch);
        train_trg_minibatch.push_back(temp_train_trg_minibatch);
        auto num_minibatches = temp_train_src_minibatch.size();
        //--------------
        ostringstream os;
        os << "Task_" << i;
        DataIter data_iter(num_minibatches, os.str());
        data_iter.idx_shuffle();
        train_ids_minibatch.push_back(data_iter);
    }
    //cerr << "zz22" << endl;
    //shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *rndeng);
    //cerr << "shuffeled. The size of the training set in minibatches is: " << train_ids_minibatch.size() << endl;

    // set up optimizers
    std::shared_ptr <Trainer> p_sgd(nullptr);
    if (opt_type == 1) {
        p_sgd.reset(new MomentumSGDTrainer(model, eta[0]));
    } else if (opt_type == 2) {
        p_sgd.reset(new AdagradTrainer(model, eta[0]));
    } else if (opt_type == 3) {
        p_sgd.reset(new AdadeltaTrainer(model, eta[0]));
    } else if (opt_type == 4) {
        p_sgd.reset(new AdamTrainer(model, eta[0]));
    } else if (opt_type == 5) {
        p_sgd.reset(new RMSPropTrainer(model, eta[0]));
    } else if (opt_type == 0) {//Vanilla SGD trainer
        p_sgd.reset(new SimpleSGDTrainer(model, eta[0]));
    } else
        assert("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");

    p_sgd->clip_threshold = g_clip_threshold; // * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching???

    //cerr << "z00\n";
    double best_loss = 9e+99;
    vector<double> last_epoch_loss(num_tasks, best_loss), current_epoch_loss(num_tasks, best_loss);
    for (auto k = 0; k < seq2seq.size(); k++) {
        seq2seq[k]->Enable_Dropout();
    }
    //p_sgd->eta_decay = eta_decay;
    //cerr << "z10\n";
    // start the multi-task learning algorithm
    unsigned long treport = 0, dreport = 0;
    float tobj = .0;
    vector<float> task_tobj(num_tasks, .0);
    vector<unsigned> task_ttokens(num_tasks, 0);
    Timer timer_epoch("completed in"), timer_iteration("completed in");

    ModelStats tstats;
    //cerr << "z20\n";
    while (p_sgd->epoch < max_epoch)// simple stopping criterion
    {
        ModelStats ctstats;
        vector<unsigned> task_list;
        if (adaptation){ // info: in the adaptation we just train for MT task
            task_list.push_back(0);
            //schedule = 100; // it is an independent schedule with a number greater than 4 ( to train epoch by epoch not MB by MB)
        }else {
            if ((schedule == 0) || (schedule == 1)) { //all tasks
                for (auto k = num_tasks - 1; k >= 0; k--) task_list.push_back(k);
                //randomised ordering of all tasks
                if (schedule == 0) shuffle(task_list.begin(), task_list.end(), *rndeng);
                //randomised ordering of the other tasks - the main task is ALWAYS last
                if (schedule == 1) shuffle(task_list.begin(), --(task_list.end()), *rndeng);
            } else if ((schedule == 2) || (schedule == 3)) { // only two tasks
                if (num_tasks > 1) {
                    auto other_task = (rand() % (num_tasks - 1)) + 1; // a random number between [1 and num_tasks-1]
                    task_list.push_back(other_task); //main task
                }
                task_list.push_back(0); //main task
                //randomised ordering of the two tasks
                if (schedule == 2) shuffle(task_list.begin(), task_list.end(), *rndeng);
                //else schedule==3 the main task is always last
            } else if (schedule == 4) { // info: go over tasks with the order of linguistic hierarchy
                for (auto k = num_tasks - 1; k >= 0; k--)
                    task_list.push_back(k); // consider all tasks with REVERSE order
            } else {
                cerr << "the schedule is not supported." << endl;
                abort();
            }
        }

        bool completed_t0_epoch = false;
        for (auto task_idx : task_list) {
            if (schedule>=4)
                cerr<< "training an epoch for task "<< std::to_string(task_idx)<<endl;
            while (true) {
                bool completed_epoch = train_ids_minibatch[task_idx].at_final();
                auto batch_idx = train_ids_minibatch[task_idx].idx_next();
                unsigned model_index = (enc_shared && dec_shared) ? 0 : task_idx;
                seq2seq[model_index]->Enable_Dropout();// enable dropout
                {
                    ComputationGraph cg;
                    Expression i_xent = seq2seq[model_index]->BuildGraph_Batch(train_src_minibatch[task_idx][batch_idx],
                                                                               train_trg_minibatch[task_idx][batch_idx],
                                                                               cg,
                                                                               ctstats, nullptr, nullptr);

                    auto loss = gamma[task_idx] *
                                i_xent; ///(train_cor[corp_idx].size()+.0); // to take into account the weight importance of different tasks
                    auto temp_obj = as_scalar(cg.forward(loss));
                    tobj += temp_obj;
                    task_tobj[task_idx] += temp_obj;
                    task_ttokens[task_idx] += ctstats.words_tgt;
                    //cout << "im here ZZ 11\n";
                    cg.backward(loss);
                    //cout << "im here ZZ 22\n";
                    p_sgd->eta = eta[task_idx];
                    p_sgd->update();
                    //cout << "im here ZZ 33\n";
                }
                if (completed_epoch) {
                    if (train_ids_minibatch[task_idx].get_epochs() >= lr_epochs) {
                        if (current_epoch_loss[task_idx] > last_epoch_loss[task_idx]) {
                            eta[task_idx] /= lr_eta_decay[task_idx];
                            cerr << ".... Task_" << task_idx << "_" << eta[task_idx] << "_down\n";
                        }
                    } else {
                        cerr << ".... Task_" << task_idx << "_" << eta[task_idx] << "_unchanged\n";
                    }
                    if (task_idx == 0) {
                        completed_t0_epoch = true;
                        cerr << "***Epoch " << p_sgd->epoch << " is finished. ";
                        cerr << "   eta: ";
                        p_sgd->update_epoch(); // increment the epoch counter
                        for (auto k = 0; k < eta.size(); k++)
                            cerr << eta[k] << " ";
                        last_epoch_loss = current_epoch_loss;
                        timer_epoch.show();
                        timer_epoch.reset();
                        cerr << ".... saving the latest epoch model ...\n";
                        //dynet::save_dynet_model("last_epoch.model", &model);
                        dynet::save_dynet_model(modfile + ".last_epoch", &model);
                    }
                    break; // info: training an epoch for the current task is completed
                }
                if (schedule < 4) // info: other schedules should not train epochs for tasks separately
                    break;
                if (schedule >= 4){ // info : we can see the effect of training other task on the main one
                    treport += 1;
                    dreport += 1;
                    if (treport == treport_every) {
                        treport = 0;
                        cerr << "[epoch=" << p_sgd->epoch + train_ids_minibatch[0].get_id() / (double) train_ids_minibatch[0].size()
                             << "] train (loss): "
                             << tobj << "   individual (loss:pplx): ";
                        tobj = 0;
                        for (auto i = 0; i < num_tasks; i++) {
                            cerr << task_tobj[i] / task_ttokens[i] << ":" << exp(task_tobj[i] / task_ttokens[i]) << " ";
                            task_tobj[i] = 0;
                            task_ttokens[i] = 0;
                        }
                        timer_iteration.show();
                        timer_iteration.reset();
                    }
                }

            }
        }
        treport += 1;
        dreport += 1;
        if (treport == treport_every) {
            treport = 0;
            cerr << "[epoch=" << p_sgd->epoch + train_ids_minibatch[0].get_id() / (double) train_ids_minibatch[0].size()
                 << "] train (loss): "
                 << tobj << "   individual (loss:pplx): ";
            tobj = 0;
            for (auto i = 0; i < num_tasks; i++) {
                cerr << task_tobj[i] / task_ttokens[i] << ":" << exp(task_tobj[i] / task_ttokens[i]) << " ";
                task_tobj[i] = 0;
                task_ttokens[i] = 0;
            }
            timer_iteration.show();
            timer_iteration.reset();
        }

        //if (dreport == dreport_every) {
        if (completed_t0_epoch){
            dreport = 0;
            completed_t0_epoch = false;
            current_epoch_loss = report_dev_eval<AM_t>(model, seq2seq, enc_shared, dec_shared,
                                                       p_sgd->epoch + train_ids_minibatch[0].get_id() /
                                                                      (double) train_ids_minibatch[0].size(), eta,
                                                       gamma,
                                                       dev_cor, best_loss, modfile);
        }
    }
}

//======
ParaCorpus Read_ParaCorpus(const string &filename, unsigned slen, dynet::Dict &sdict, dynet::Dict &tdict) {
    ifstream in(filename);
    assert(in);

    ParaCorpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    while (getline(in, line)) {
        ++lc;
        Sentence source, target;
        read_sentence_pair(line, source, sdict, target, tdict);

        if (slen > 0/*length limit*/) {
            if (source.size() > slen || target.size() > slen)
                continue;// ignore this sentence
        }

        corpus.push_back(SentencePair(source, target));

        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
            (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sdict.size() << " & " << tdict.size()
         << " types\n";
    return corpus;
}

MonoCorpus Read_Corpus(const string &filename, unsigned slen, dynet::Dict &sdict, dynet::Dict &tdict) {
    ifstream in(filename);
    assert(in);

    MonoCorpus corpus;
    string line;
    int lc = 0, stoks = 0; //, ttoks = 0;
    while (getline(in, line)) {
        ++lc;
        Sentence source, target;
        //source = read_sentence(line, sdict);
        read_sentence_pair(line, source, sdict, target, tdict);

        if (slen > 0/*length limit*/) {
            if (source.size() > slen)
                continue;// ignore this sentence
        }

        corpus.push_back(Sentence(source));

        stoks += source.size();
        //ttoks += target.size();

        if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << stoks << " tokens (s), " << sdict.size() << " types\n";
    return corpus;
}
