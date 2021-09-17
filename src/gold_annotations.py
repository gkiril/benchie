#        <NAME OF THE PROGRAM THIS FILE BELONGS TO>
# 	  
#   File:     gold_annotations.py
#   Authors:  Kiril Gashteovski (kiril.gashteovski@neclab.eu) 
#             Mingying Yu (mingying.yu@neclab.eu)
#             Bhushan Kotnis (bhushan.kotnis@neclab.eu)
#             Carolin Lawrence (carolin.lawrence@neclab.eu)
#             Goran Glavaš (goran@informatik.uni-mannheim.de)
#             Mathias Niepert (mathias.niepert@neclab.eu)
# 
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.  
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
# 
#        PROPRIETARY INFORMATION ---  
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.  
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

import re
import itertools

import numpy as np

from utils import triple2string

class GoldAnnotations:
    def __init__(self):
        """
            Default constructor creates empty list for the sentences ID, and empty dictionaries for the input sentences and 
            the golden annotations
        """
        # Store sentence IDs written in the golden annotation input text file
        self.sent_ids = [] 

        # Store sentences as a dictionary, where key = sent id; value = sentence as a string
        self.sentences = {}

        # Store the golden annotations in a dictionary, where key = sent id; value = list of lists, where each list is 
        # a triple synset, and each element is a triple
        self.golden_annotations = {}

    def load_gold_annotations(self, filename: str, load_mode: str = "full"):
        """
            Loads the input sentences IDs (in self.sent_ids), input sentences (in self.sentences) and the dictionary of golden 
            annotations (in self.gold_annotations) from the golden annotations written in filename.

            Args
            ----
            filename: str
                The name of the file where the golden annotations are written
            load_mode: str
                The mode of loading the data. There are only two possible options: "full" or "minimal" ("full" is the default).
                In "full" mode, everything in the golden annotations is loaded. In "minimal" mode, only the tokens that are not optional
                are loaded (the optional tokens are omitted). 
        """

        # Load the sentences
        self.__load_sentences(filename)
        
        # list of extraction annotations
        extraction_annotations = self.__get_extraction_annotations(filename)
        
        # cluster gold annotations into sentence-level.
        # Output a list: ['1--> Cluster 1:', ['He', 'served as', '[the] [first] Prime Minister [of Australia]'], ['He', 'served', 'as [the] [first] Prime Minister [of Australia]'], '1--> Cluster 2:', ... 
        extraction_annotations_per_sentence = self.__get_extraction_annotations_per_sentence(extraction_annotations)

        if load_mode == "full":
            # Create a gold dictionary, where the key is the sentence ID, the value is a list (i.e., a cluster, a triple synset) of golden extractions 
            self.golden_annotations = self.__generate_golden_extractions(extraction_annotations_per_sentence)
            self.__generate_golden_extractions_from_optional_tokens(self.golden_annotations)
        elif load_mode == "minimal":
            for sentence in extraction_annotations_per_sentence:
                for entry in sentence:
                    if type(entry) is list:
                        triple = entry
                        for i in range(len(triple)):
                            if "[" in triple[i]:
                                # Remove tokens/phrases in square brackets
                                triple[i] = re.sub("[\(\[].*?[\)\]]", "", triple[i]).strip()

                                # Replace multiple spaces with one
                                triple[i] = re.sub(' +', ' ', triple[i])
            
            # Create a gold dictionary, where the key is the sentence ID, the value is a list (i.e., a cluster, a triple synset) of golden extractions 
            self.golden_annotations = self.__generate_golden_extractions(extraction_annotations_per_sentence)
            self.__generate_golden_extractions_from_optional_tokens(self.golden_annotations)            
        else:
            raise Exception

    def __load_sentences(self, filename: str):
        """
            Loads the input sentences (in self.sentences) and their IDs (in self.sent_ids) from the golden annotations written in filename

            Args
            ----
            filename: str
                The name of the file where the golden annotations are written
        """

        with open(filename,'r',encoding='UTF-8') as f:
            file_lines = [line.strip() for line in f]

        for i in range(0, len(file_lines)):
            line = file_lines[i]

            if "sent_id:" in line:
                sent_id = line.split("\t")[0].split("sent_id:")[1]
                sentence = line.split("\t")[1]
                if sent_id in self.sent_ids:
                    raise Exception
                self.sent_ids.append(sent_id)
                self.sentences[sent_id] = sentence

    def __get_extraction_annotations(self, filename: str) -> list:
        """
            Get a list of file lines where the gold annotations are written (i.e., filter out the file lines where there are either 
            sentences or empty lines).

            Args
            ----
            filename: str
                The name of the file where the golden annotations are written

            Returns
            -------
            gold: list 
                list of elements. Each element is either a string (triple cluster name) or a list of slots (subj, rel, obj). 
                The triple cluster name is written in the format '1--> Cluster 2:' where 1 is the sent ID and 2 is the cluster ID.
                The list of slots is a triple, where each element is a string. The tokens in square brackets signify optional tokens.
        """

        with open(filename,'r',encoding='UTF-8') as f:
            file_lines = [line.strip() for line in f]
        
        gold = []
        for line in file_lines:
            if ' --> ' in line:
                gold.append(re.split(r' --> ', line))   #add extraction line
            elif "sent_id:" in line:
                continue    #remove sentence line
            else:
                gold.append(line)   # add cluster order
        
        return gold

    def __get_extraction_annotations_per_sentence(self, extraction_annotations: list) -> list:
        """
            Args
            ----
            extraction_annotations: list
                List of extraction annotations, which were obtained with the __get_extraction_annotations function. For details,
                see the explanation for the returned variable gold in the function __get_extraction_annotations.

            Returns
            -------
            extractions_per_sentence: list
                List that contains details about the golden annotations (e.g., cluster and triples). The number of the elements of the 
                list corresponds to the number of input sentences. In other words, this function restructures the data returned from 
                the function __get_extraction_annotations
                
        """

        extractions_per_sentence=[]
        length=0
        for i in range(len(extraction_annotations)):
            if extraction_annotations[i]=='':
                sublist=extraction_annotations[length:i]
                length=len(sublist)
                extractions_per_sentence.append(sublist)
                length=i+1
            if i==len(extraction_annotations)-1:
                sublist=extraction_annotations[length:]
                extractions_per_sentence.append(sublist)
            else:
                continue
        
        return extractions_per_sentence

    def __generate_golden_extractions(self, gold_sentence_level: list) -> dict:
        """
            Create a gold dictionary, where the key is the sentence ID, the value is a list (i.e., a cluster, a triple synset) of 
            golden extractions. Tokens written with square brackets are optional.

            Args
            ----
                gold_sentence_level: list

            Returns
            -------
                gold_dict: dict
                    key: sent id; value: list (cluster) of extractions; in each slot, the tokens in square brackets are optional

        """
        gold_cluster_level = []

        for s in gold_sentence_level:
            cluster=[]
            length=0
            for i in range(1,len(s)):
                if 'Cluster' in s[i]:
                    sublist=s[length:i]
                    length=len(sublist)
                    cluster.append(sublist)
                    length=i
                if i==len(s)-1:
                    sublist=s[length:]
                    cluster.append(sublist)
                else:
                    continue
            for i in cluster:
                for j in i:
                    if 'Cluster' in j:
                        i.remove(j)
            gold_cluster_level.append(cluster)

        gold_dict={}

        for (i, j) in zip(self.sent_ids, gold_cluster_level):
            gold_dict[i]=j
        
        return gold_dict

    def __generate_golden_extractions_from_optional_tokens(self, gold_dict: dict):
        """
            Generate all possible golden extractions from the optional tokens (in square brackets). Save results in gold_dict.

            Args 
            ----
                gold_dict: key: sent id; value: list (cluster) of extractions; in each slot, the tokens in square brackets are optional.
                            Once the function is executed, the changes are stored in gold_dict.

        """

        for sentence in gold_dict:
            for cluster in gold_dict[sentence]:
                gold_annotation=[]
                for triple in cluster:
                    sublist=[]
                    for slot in triple:
                        new_slot=re.split(r' ',slot)
                        new_list=[]
                        it=iter(range(len(new_slot)))
                        for i in it:
                            if '[' not in new_slot[i] and ']' not in new_slot[i]:
                                new_list.append(new_slot[i])
                            elif '[' in new_slot[i] and ']' in new_slot[i]:
                                new_list.append(new_slot[i])
                            elif '[' in new_slot[i]:
                                for j in range(i+1,len(new_slot)):
                                    if ']' in new_slot[j]:
                                        index=j
                                        new_string=' '.join(new_slot[i:index+1])
                                        new_list.append(new_string)
                                        for times in range(int(index-i-1)):
                                            next(it)
                                        break
                        sublist.append(new_list)
                    gold_annotation.append(sublist)
                cluster.clear()
                for c in gold_annotation:
                    cluster.append(c)
                    
        for sentence in gold_dict:
            for cluster in gold_dict[sentence]:
                cluster_annotation=[]
                for triple in cluster:
                    sub,pred,obj=triple[0:3]
                    g_subject=self.__generate_slots_from_optional_tokens(sub)
                    g_predicate=self.__generate_slots_from_optional_tokens(pred)
                    g_object=self.__generate_slots_from_optional_tokens(obj)
                    #new triple form
                    t=[]
                    t.append(g_subject)
                    t.append(g_predicate)
                    t.append(g_object)
                    a=list(itertools.product(*t))
                    for i in a:
                        cluster_annotation.append(i)
                cluster.clear()
                for j in cluster_annotation:
                    cluster.append(j)
                    
        # remove brackets in the triple
        for sentence in gold_dict:
            for cluster in gold_dict[sentence]:
                for triple in range(len(cluster)):
                    new_triple=[]
                    for slot in range(len(cluster[triple])):
                        slot_merge=' '.join(cluster[triple][slot])   #merge words in every slot into a single string
                        new_slot=re.sub('\\[|\\]', '', slot_merge)   #delete brackets
                        new_triple.append(new_slot)
                    cluster[triple]=new_triple
    
    def __generate_slots_from_optional_tokens(self, slot: list) -> list:
        """
            Given a slot (a list of tokens) with potentially optional tokens (in square brackets), generate all possible combinations
            of the surface representation of the slot.

            Args
            ----
                slot: list
                    List of tokens with potentially optional tokens (in square brackets)

            Returns
            -------
                sub_all: list
                    List of all possible surface realizations of the slot.
        """
        index=[]
        for i in range(len(slot)):
            if '[' in slot[i]:
                index.append(i)
        j=[]
        for i in range(1, len(index)+1):
            els = [list(x) for x in itertools.combinations(index, i)]
            j.extend(els)
        for i in j:
            i.reverse()
        sub_all=[]
        sub_all.append(slot)
        
        for i in j:
            p=slot.copy()
            for j in i:
                del p[j]
            sub_all.append(p)
        
        return sub_all

    def compute_stats(self) -> dict:
        """
            Compute certain stats about the golden annotations. 

            Returns:
            --------
                stats: dict
                    Dictionary, where the key is the name of the statistics; value is the value of that particular statistic
                    A list of all possible stats:
                    * triple_synset_count: Total number of triple synset
                    * extractions_count: Total number of extractions
                    * extractions_per_sentence: Number of extractions per sentence
                    * triple_synsets_per_sentence: Number of triple synsets per sentence
                    * extractions_per_triple_synset: Number of extractions per triple synset
                    * avg_extraction_length: Avg. length of extraction
        """
        stats = {}
        
        triple_synset_count = 0
        extractions_count = 0
        sent_count = len(self.sentences)
        extraction_length = []

        for sent_id in self.golden_annotations:
            triple_synset_count += len(self.golden_annotations[sent_id])
            for synset in self.golden_annotations[sent_id]:
                extractions_set = set()
                for triple in synset:
                    triple_length = 0
                    triple_length += len(triple[0].split(" "))
                    triple_length += len(triple[1].split(" "))
                    triple_length += len(triple[2].split(" "))
                    extraction_length.append(triple_length)
                    extractions_set.add(triple2string(triple))
                extractions_count += len(extractions_set)
        
        stats['triple_synset_count'] = triple_synset_count
        stats['extractions_count'] = extractions_count
        stats['extractions_per_sentence'] = extractions_count/sent_count
        stats['triple_synsets_per_sentence'] = triple_synset_count/sent_count
        stats['extractions_per_triple_synset'] = extractions_count/triple_synset_count
        stats['avg_extraction_length'] = np.mean(extraction_length)

        return stats