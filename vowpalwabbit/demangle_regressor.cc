/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */

#include "reductions.h"
#include "interactions.h"
#include "parse_args.h"
#include "vw.h"

struct demangle_data
{
    vw* all;
    size_t offset;
    size_t class_cnt;
    vector<string> ns_pre;
    io_buf* out_file;
    size_t ft_found;
};

inline void demangle_interaction(demangle_data& dat, const audit_data* f)
{ // same as audit_interaction in gd.cc
    if (f == nullptr)
    {
        dat.ns_pre.pop_back();
        return;
    }

    string ns_pre;
//    if (!dat.ns_pre.empty())
//        ns_pre += '*';

    if (f->space && (*(f->space) != ' '))
    {
        std::string result;
            std::stringstream ss;
            ss  << "\\x" << std::hex <<+(unsigned char)*(f->space);
            ss >> result;
        ns_pre.append(result);
//        ns_pre += '^';
    }
//    ns_pre.append(f->feature);
    dat.ns_pre.push_back(ns_pre);
}

inline float sign(float w){ if (w < 0.) return -1.; else  return 1.;}
inline float trunc_weight(const float w, const float gravity){
   return (gravity < fabsf(w)) ? w - sign(w) * gravity : 0.f;
 }

inline void demangle_feature(demangle_data& dat, const float /*ft_weight*/, const uint32_t ft_idx)
{
    size_t index = ft_idx & dat.all->reg.weight_mask;
    weight* weights = dat.all->reg.weight_vector;
    size_t stride_shift = dat.all->reg.stride_shift;

    string ns_pre;
    for (vector<string>::const_iterator s = dat.ns_pre.begin(); s != dat.ns_pre.end(); ++s) ns_pre += *s;

    if (weights[index] == 0) return;
    else ++dat.ft_found;

    ostringstream tempstream;
    tempstream << ',' /*<< (index >> stride_shift) << ':'*/ << trunc_weight(weights[index], (float)dat.all->sd->gravity) * (float)dat.all->sd->contraction;

//    if(dat.all->adaptive)
//        tempstream << '@' << weights[index+1];


    string temp = ns_pre+tempstream.str()+'\n';
    bin_text_write(*dat.out_file, nullptr, 0, temp.c_str(), temp.size(), true );

    weights[index] = 0.;
}

// This is a learner which does nothing with examples.
//void learn(demangle_data&, LEARNER::base_learner&, example&) {}

void demangle(demangle_data& dd, LEARNER::base_learner& /*base*/, example& ec)
{
    size_t c = 0;
    while ( c < dd.class_cnt )
    {

    dd.offset = ec.ft_offset;
    if (dd.class_cnt > 0)
    {   // add class prefix for multiclass problems
        std::stringstream ss;
//        ss << c << ':';
        dd.ns_pre.push_back(ss.str());
    }

    for (unsigned char* i = ec.indices.begin; i != ec.indices.end; ++i)
    {
        v_array<audit_data>& ns =  ec.audit_features[(size_t)*i];
        for (audit_data* a = ns.begin; a != ns.end; ++a)
        {
            demangle_interaction(dd, a);
            demangle_feature(dd, a->x, (uint32_t)a->weight_index + ec.ft_offset);
            demangle_interaction(dd, NULL);
        }
    }


    INTERACTIONS::generate_interactions<demangle_data, const uint32_t, demangle_feature, audit_data, demangle_interaction >(*dd.all, ec, dd, ec.audit_features);

    ec.ft_offset += 4*(++c);

    if (dd.class_cnt > 0) dd.ns_pre.pop_back();

    }

    ec.ft_offset -= 4*c;
}

void end_examples(demangle_data& d)
{
    d.out_file->flush(); // close_file() should do this for me ...
    d.out_file->close_file();
    delete (d.out_file);
    d.out_file = NULL;
}

void finish_example(vw& all, demangle_data& dd, example& ec)
{

    #define print_ex(cnt, found, prog) { \
    std::cerr << std::left   \
          << std::setw(shared_data::col_example_counter) << cnt   \
          << " " << std::right   \
          << std::setw(9) << found   \
          << " "  << std::right  \
          << std::setw(12) << prog << '%'   \
          << std::endl; }

    bool printed = false;
    if (ec.example_counter+1 >= all.sd->dump_interval && !all.quiet)
    {
        print_ex(ec.example_counter+1, dd.ft_found, dd.ft_found*100/all.weights_loaded);
        all.sd->weighted_examples = ec.example_counter+1; //used in update_dump_interval
        all.sd->update_dump_interval(all.progress_add, all.progress_arg);
        printed = true;
    }

    if (dd.ft_found == all.weights_loaded)
    {
        if (!printed)
            print_ex(ec.example_counter+1, dd.ft_found, 100);
         set_done(all);
    }
    VW::finish_example(all, &ec);
}

inline void set_class_cnt(demangle_data& d)
{
    po::variables_map& vm = d.all->vm;
    if (vm.count("oaa"))
        d.class_cnt = vm["oaa"].as<size_t>();
    else
        if (vm.count("ect"))
            d.class_cnt = vm["ect"].as<size_t>();
        else
            if (vm.count("csoaa"))
                d.class_cnt = vm["csoaa"].as<size_t>();
            else
                d.class_cnt = 1;
}

void init_driver(demangle_data& dd)
{
    po::variables_map& vm = dd.all->vm;
    if ( (vm.count("cache_file") || vm.count("cache") ) && !vm.count("kill_cache") )
        THROW("--demangle_regressor can't be used with dataset's cache file.");

    if (dd.all->weights_loaded == 0)
        THROW("Regressor has no non-zero weights. Nothing to demangle.");

    set_class_cnt(dd);
}



LEARNER::base_learner* demangle_setup(vw& all)
{
    if (missing_option<string,false>(all, "demangle_regressor", "restore feature names from regressor hashes using same dataset")) return nullptr;

    all.demangle_reg = true;
    po::variables_map& vm = all.vm;

    string out_file = vm["demangle_regressor"].as<string>();
    if (out_file.empty())    
        THROW("--demangle_regressor argument is missing.");

    if (all.numpasses > 1)
        THROW("--demangle_regressor can't be used with --passes > 1.");

    all.audit = true;

    demangle_data& d = calloc_or_throw<demangle_data>();
    d.all = &all;
    d.out_file = new io_buf();
    d.out_file->open_file( out_file.c_str(), all.stdin_off, io_buf::WRITE );


    LEARNER::learner<demangle_data>& ret = LEARNER::init_learner<demangle_data>(&d, setup_base(all), demangle, demangle, 1);
//    ret.set_predict(demangle);
    ret.set_end_examples(end_examples);
    ret.set_finish_example(finish_example);
    ret.set_init_driver(init_driver);



    return LEARNER::make_base<demangle_data>(ret);
}
