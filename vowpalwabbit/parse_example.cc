 /*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */

#include <math.h>
#include <ctype.h>
#include "parse_example.h"
#include "hash.h"
#include "unique_sort.h"
#include "global_data.h"
#include "constant.h"

using namespace std;

char* copy(char* base)
{ size_t len = 0;
  while (base[len++] != '\0');
  char* ret = calloc_or_throw<char>(len);
  memcpy(ret,base,len);
  return ret;
}

template<bool audit>
class TC_parser
{

public:
  char* beginLine;
  char* reading_head;
  char* endLine;
  float cur_channel_v;
  bool  new_index;
  size_t anon;
  size_t channel_hash;
  char* base;
  unsigned char index;
  float v;
  bool redefine_some;
  unsigned char (*redefine)[256];
  parser* p;
  example* ae;
  uint32_t* affix_features;
  bool* spelling_features;
  v_array<char> spelling;

  vector<feature_dict*>* namespace_dictionaries;

  ~TC_parser() { }

  inline void parserWarning(const char* message, char* begin, char* pos, const char* message2)
  { cout << message << std::string(begin, pos - begin).c_str() << message2
         << "in Example #" << this->p->end_parsed_examples << ": \"" << std::string(this->beginLine, this->endLine).c_str() << "\""
         << endl;
  }

  inline float featureValue()
  { if(*reading_head == ' ' || *reading_head == '\t' || *reading_head == '|' || reading_head == endLine || *reading_head == '\r')
      return 1.;
    else if(*reading_head == ':')
    { // featureValue --> ':' 'Float'
      ++reading_head;
      char *end_read = nullptr;
      v = parseFloat(reading_head,&end_read);
      if(end_read == reading_head)
      { parserWarning("malformed example! Float expected after : \"", beginLine, reading_head, "\"");
      }
      if(nanpattern(v))
      { v = 0.f;
        parserWarning("warning: invalid feature value:\"", reading_head, end_read, "\" read as NaN. Replacing with 0.");
      }
      reading_head = end_read;
      return v;
    }
    else
    { // syntax error
      parserWarning("malformed example! '|', ':', space, or EOL expected after : \"", beginLine, reading_head, "\"");
      return 0.f;
    }
  }

  inline substring read_name()
  { substring ret;
    ret.begin = reading_head;
    while( !(*reading_head == ' ' || *reading_head == ':' || *reading_head == '\t' || *reading_head == '|' || reading_head == endLine || *reading_head == '\r' ))
      ++reading_head;
    ret.end = reading_head;

    return ret;
  }

  inline void maybeFeature()
  { if(*reading_head == ' ' || *reading_head == '\t' || *reading_head == '|'|| reading_head == endLine || *reading_head == '\r' )
    { // maybeFeature --> ø
    }
    else
    { // maybeFeature --> 'String' FeatureValue
      substring feature_name=read_name();
      v = cur_channel_v * featureValue();
      size_t word_hash;
      if (feature_name.end != feature_name.begin)
        word_hash = (p->hasher(feature_name,(uint32_t)channel_hash));
      else
        word_hash = channel_hash + anon++;
      if(v == 0) return; //dont add 0 valued features to list of features
      feature f = {v,(uint32_t)word_hash };
      ae->sum_feat_sq[index] += v*v;
      ae->atomics[index].push_back(f);
      if(audit)
      { v_array<char> feature_v = v_init<char>();
        push_many(feature_v, feature_name.begin, feature_name.end - feature_name.begin);
        feature_v.push_back('\0');
        audit_data ad = {copy(base),feature_v.begin,word_hash,v,true};
        ae->audit_features[index].push_back(ad);
      }
      if ((affix_features[index] > 0) && (feature_name.end != feature_name.begin))
      { if (ae->atomics[affix_namespace].size() == 0)
          ae->indices.push_back(affix_namespace);
        uint32_t affix = affix_features[index];
        while (affix > 0)
        { bool is_prefix = affix & 0x1;
          uint32_t len   = (affix >> 1) & 0x7;
          substring affix_name = { feature_name.begin, feature_name.end };
          if (affix_name.end > affix_name.begin + len)
          { if (is_prefix)
              affix_name.end = affix_name.begin + len;
            else
              affix_name.begin = affix_name.end - len;
          }
          word_hash = p->hasher(affix_name,(uint32_t)channel_hash) * (affix_constant + (affix & 0xF) * quadratic_constant);
          feature f2 = { v, (uint32_t) word_hash };
          ae->sum_feat_sq[affix_namespace] += v*v;
          ae->atomics[affix_namespace].push_back(f2);
          if (audit)
          { v_array<char> affix_v = v_init<char>();
            if (index != ' ') affix_v.push_back(index);
            affix_v.push_back(is_prefix ? '+' : '-');
            affix_v.push_back('0' + len);
            affix_v.push_back('=');
            push_many(affix_v, affix_name.begin, affix_name.end - affix_name.begin);
            affix_v.push_back('\0');
            audit_data ad = {copy((char*)"affix"),affix_v.begin,word_hash,v,true};
            ae->audit_features[affix_namespace].push_back(ad);
          }

          affix >>= 4;
        }
      }
      if (spelling_features[index])
      { if (ae->atomics[spelling_namespace].size() == 0)
          ae->indices.push_back(spelling_namespace);
        //v_array<char> spelling;
        spelling.erase();
        for (char*c = feature_name.begin; c!=feature_name.end; ++c)
        { char d = 0;
          if      ((*c >= '0') && (*c <= '9')) d = '0';
          else if ((*c >= 'a') && (*c <= 'z')) d = 'a';
          else if ((*c >= 'A') && (*c <= 'Z')) d = 'A';
          else if  (*c == '.')                 d = '.';
          else                                 d = '#';
          //if ((spelling.size() == 0) || (spelling.last() != d))
          spelling.push_back(d);
        }
        substring spelling_ss = { spelling.begin, spelling.end };
        size_t word_hash = hashstring(spelling_ss, (uint32_t)channel_hash);
        feature f2 = { v, (uint32_t) word_hash };
        ae->sum_feat_sq[spelling_namespace] += v*v;
        ae->atomics[spelling_namespace].push_back(f2);
        if (audit)
        { v_array<char> spelling_v = v_init<char>();
          if (index != ' ') { spelling_v.push_back(index); spelling_v.push_back('_'); }
          push_many(spelling_v, spelling_ss.begin, spelling_ss.end - spelling_ss.begin);
          spelling_v.push_back('\0');
          audit_data ad = {copy((char*)"spelling"),spelling_v.begin,word_hash,v,true};
          ae->audit_features[spelling_namespace].push_back(ad);
        }
      }
      if (namespace_dictionaries[index].size() > 0)
      { for (size_t dict=0; dict<namespace_dictionaries[index].size(); dict++)
        { feature_dict* map = namespace_dictionaries[index][dict];
          uint32_t hash = uniform_hash(feature_name.begin, feature_name.end-feature_name.begin, quadratic_constant);
          v_array<feature>* feats = map->get(feature_name, hash);
          if ((feats != nullptr) && (feats->size() > 0))
          { if (ae->atomics[dictionary_namespace].size() == 0)
              ae->indices.push_back(dictionary_namespace);
            push_many(ae->atomics[dictionary_namespace], feats->begin, feats->size());
            for (feature*f = feats->begin; f != feats->end; ++f)
              ae->sum_feat_sq[dictionary_namespace] += f->x * f->x;
            if (audit)
            { for (feature*f = feats->begin; f != feats->end; ++f)
              { uint32_t id = f->weight_index;
                size_t len = 2 + (feature_name.end-feature_name.begin) + 1 + (size_t)ceil(log10(id)) + 1;
                char* str = calloc_or_throw<char>(len);
                str[0] = index;
                str[1] = '_';
                char *c = str+2;
                for (char* fc=feature_name.begin; fc!=feature_name.end; ++fc) *(c++) = *fc;
                *(c++) = '=';
                sprintf(c, "%d", id);
                audit_data ad = { copy((char*)"dictionary"), str, f->weight_index, f->x, true };
                ae->audit_features[dictionary_namespace].push_back(ad);
              }
            }
          }
        }
      }
    }
  }

  inline void nameSpaceInfoValue()
  { if(*reading_head == ' ' || *reading_head == '\t' || reading_head == endLine || *reading_head == '|' || *reading_head == '\r' )
    { // nameSpaceInfoValue -->  ø
    }
    else if(*reading_head == ':')
    { // nameSpaceInfoValue --> ':' 'Float'
      ++reading_head;
      char *end_read = nullptr;
      cur_channel_v = parseFloat(reading_head,&end_read);
      if(end_read == reading_head)
      { parserWarning("malformed example! Float expected after : \"", beginLine, reading_head, "\"");
      }
      if(nanpattern(cur_channel_v))
      { cur_channel_v = 1.f;
        parserWarning("warning: invalid namespace value:\"", reading_head, end_read, "\" read as NaN. Replacing with 1.");
      }
      reading_head = end_read;
    }
    else
    { // syntax error
      parserWarning("malformed example! '|',':', space, or EOL expected after : \"", beginLine, reading_head, "\"");
    }
  }

  inline void nameSpaceInfo()
  { if(reading_head == endLine ||*reading_head == '|' || *reading_head == ' ' || *reading_head == '\t' || *reading_head == ':' || *reading_head == '\r')
    { // syntax error
      parserWarning("malformed example! String expected after : \"", beginLine, reading_head, "\"");
    }
    else
    { // NameSpaceInfo --> 'String' NameSpaceInfoValue
      index = (unsigned char)(*reading_head);
      if (redefine_some) index = (*redefine)[index]; //redefine index
      if(ae->atomics[index].begin == ae->atomics[index].end)
        new_index = true;
      substring name = read_name();
      if(audit)
      { v_array<char> base_v_array = v_init<char>();
        push_many(base_v_array, name.begin, name.end - name.begin);
        base_v_array.push_back('\0');
        if (base != nullptr)
          free(base);
        base = base_v_array.begin;
      }
      channel_hash = p->hasher(name, hash_base);
      nameSpaceInfoValue();
    }
  }

  inline void listFeatures()
  { while(*reading_head == ' ' || *reading_head == '\t')
    { //listFeatures --> ' ' MaybeFeature ListFeatures
      ++reading_head;
      maybeFeature();
    }
    if(!(*reading_head == '|' || reading_head == endLine || *reading_head == '\r'))
    { //syntax error
      parserWarning("malformed example! '|',space, or EOL expected after : \"", beginLine, reading_head, "\"");
    }
  }

  inline void nameSpace()
  { cur_channel_v = 1.0;
    index = 0;
    new_index = false;
    anon = 0;
    if(*reading_head == ' ' || *reading_head == '\t' || reading_head == endLine || *reading_head == '|' || *reading_head == '\r' )
    { // NameSpace --> ListFeatures
      index = (unsigned char)' ';
      if(ae->atomics[index].begin == ae->atomics[index].end)
        new_index = true;
      if(audit)
      { if (base != nullptr)
          free(base);
        base = calloc_or_throw<char>(2);
        base[0] = ' ';
        base[1] = '\0';
      }
      channel_hash = 0;
      listFeatures();
    }
    else if(*reading_head != ':')
    { // NameSpace --> NameSpaceInfo ListFeatures
      nameSpaceInfo();
      listFeatures();
    }
    else
    { // syntax error
      parserWarning("malformed example! '|',String,space, or EOL expected after : \"", beginLine, reading_head, "\"");
    }
    if(new_index && ae->atomics[index].begin != ae->atomics[index].end)
      ae->indices.push_back(index);
  }

  inline void listNameSpace()
  { while(*reading_head == '|')   // ListNameSpace --> '|' NameSpace ListNameSpace
    { ++reading_head;
      nameSpace();
    }
    if(reading_head != endLine && *reading_head != '\r')
    { // syntax error
      parserWarning("malformed example! '|' or EOL expected after : \"", beginLine, reading_head, "\"");
    }
  }

  TC_parser(char* reading_head, char* endLine, vw& all, example* ae)
  { spelling = v_init<char>();
    if (endLine != reading_head)
    { this->beginLine = reading_head;
      this->reading_head = reading_head;
      this->endLine = endLine;
      this->p = all.p;
      this->redefine_some = all.redefine_some;
      this->redefine = &all.redefine;
      this->ae = ae;
      this->affix_features = all.affix_features;
      this->spelling_features = all.spelling_features;
      this->namespace_dictionaries = all.namespace_dictionaries;
      this->base = nullptr;
      listNameSpace();
      if (base != nullptr)
        free(base);
    }
  }
};

void substring_to_example(vw* all, example* ae, substring example)
{ all->p->lp.default_label(&ae->l);
  char* bar_location = safe_index(example.begin, '|', example.end);
  char* tab_location = safe_index(example.begin, '\t', bar_location);
  substring label_space;
  if (tab_location != bar_location)
  { label_space.begin = tab_location + 1;
  }
  else
  { label_space.begin = example.begin;
  }
  label_space.end = bar_location;

  if (*example.begin == '|')
  { all->p->words.erase();
  }
  else
  { tokenize(' ', label_space, all->p->words);
    if (all->p->words.size() > 0 && (all->p->words.last().end == label_space.end	|| *(all->p->words.last().begin) == '\'')) //The last field is a tag, so record and strip it off
    { substring tag = all->p->words.pop();
      if (*tag.begin == '\'')
        tag.begin++;
      push_many(ae->tag, tag.begin, tag.end - tag.begin);
    }
  }

  if (all->p->words.size() > 0)
    all->p->lp.parse_label(all->p, all->sd, &ae->l, all->p->words);

  if (all->audit || all->hash_inv)
    TC_parser<true> parser_line(bar_location,example.end,*all,ae);
  else
    TC_parser<false> parser_line(bar_location,example.end,*all,ae);
}

int read_features(void* in, example* ex)
{ vw* all = (vw*)in;
  example* ae = (example*)ex;
  char *line=nullptr;
  size_t num_chars_initial = readto(*(all->p->input), line, '\n');
  if (num_chars_initial < 1)
    return (int)num_chars_initial;
  size_t num_chars = num_chars_initial;
  if (line[0] =='\xef' && num_chars >= 3 && line[1] == '\xbb' && line[2] == '\xbf')
  { line += 3;
    num_chars -= 3;
  }
  if (line[num_chars-1] == '\n')
    num_chars--;
  if (line[num_chars-1] == '\r')
    num_chars--;
  substring example = {line, line + num_chars};
  substring_to_example(all, ae, example);

  return (int)num_chars_initial;
}

namespace VW
{
void read_line(vw& all, example* ex, char* line)
{ substring ss = {line, line+strlen(line)};
  while ((ss.end >= ss.begin) && (*(ss.end-1) == '\n')) ss.end--;
  substring_to_example(&all, ex, ss);
}
}
