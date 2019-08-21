import os
import json
from nltk.tree import Tree
import networkx as nx


def json_load(file_path):
    with open(file_path, 'r') as file_input:
        return json.load(file_input)

class Dep_Graph(nx.DiGraph):
    
    def __init__(self):
        super(Dep_Graph, self).__init__()
    
    def build_from_dep_edge(self, dep_edge_list):
        head_ids = []
        token_ids = []
        for edge in dep_edge_list:
            head_ids.append(edge[1])
            token_ids.append(edge[2])
        self.add_nodes_from(token_ids)
        for head_id, token_id in zip(head_ids, token_ids):
            if head_id != -1: 
                self.add_edge(head_id, token_id)
    
    def children(self, index):
        return list(self.successors(index))
    
    def parent(self, index):
        return list(self.predecessors(index))
    
    def successor(self, index):
        return nx.descendants(self, index)


class pdtb3:
    def __init__(self, path, folder_list=list(range(2,24))):
        if not os.path.exists(path):
            print('PDTB data path does not exist, please check the path')
            assert False
        self.path = path
        self.relation_data = self._load(folder_list)
        self.rel_id2docidOffset = self._build_rel_id2docidOffset_map()
        self.rel_id = list(self.rel_id2docidOffset.keys())
        self.build_index(['Sense','Type'])
    
    def _build_rel_id2docidOffset_map(self):
        rel_id2docidOffset = {}
        for folder_id in self.relation_data:
            for doc_id in self.relation_data[folder_id]:
                for offset, relation in enumerate(self.relation_data[folder_id][doc_id]):
                    rel_id2docidOffset[relation['ID']] = (doc_id, offset)
        return rel_id2docidOffset
    
    def query_rel_id(doc_id, offset):
        folder_id = self._extract_folder_id(self, doc_id)
        return self.relation_data[folder_id][doc_id][offset]['ID']
    
    def build_index(self, key_list):
        self.index = {key:{} for key in key_list}
        
        for rel_id in self.rel_id2docidOffset:
            unique_relation_identifier = rel_id
            relation = self._extract_relation(rel_id)
            for key in key_list:
                assert key in relation
                 ## multi sub-key
                if isinstance(relation[key], list):
                    for sub_key in relation[key]:
                        if sub_key not in self.index[key]:
                            self.index[key][sub_key] = [unique_relation_identifier]
                        else:
                            self.index[key][sub_key].append(unique_relation_identifier)
                else:
                    if relation[key] not in self.index[key]:
                        self.index[key][relation[key]] = [unique_relation_identifier]
                    else:
                        self.index[key][relation[key]].append(unique_relation_identifier)

    def _load(self, folder_list=list(range(2,24))):
        
        relation_data_dict = {}
        for folder_ind in folder_list:
            index = '0{}'.format(folder_ind) if len(str(folder_ind)) == 1  else str(folder_ind)
            folder_name = 'wsj_{}'.format(index) 
            relation_data_dict[folder_name] = json_load(os.path.join(self.path, folder_name +'.json'))
        return relation_data_dict
    
    def _extract_relation(self, *argv):
        assert len(argv) == 1 or len(argv) == 2
        if len(argv) == 1:
            assert argv[0] in self.rel_id2docidOffset
            doc_id, offset = self.rel_id2docidOffset[argv[0]]
        else:
            assert isinstance(argv[0], str)
            assert isinstance(argv[1], int)
            doc_id = argv[0]
            offset = argv[1]
        
        folder_id = self._extract_folder_id(doc_id)
        return self.relation_data[folder_id][doc_id][offset]
    
    def _extract_folder_id(self, doc_id):
        return doc_id[0:6]
    
    def get_raw_text(self, rel_id, Attr):
        relation = self._extract_relation(rel_id)
        return relation[Attr]['RawText']
    
    def get_connective(self, rel_id):
        relation = self._extract_relation(rel_id)
        return relation['Connective']['RawText']
    
    def get_sense(self, rel_id, level=0):
        '''
        return the sense at level 
                Expansion.Level-of-detail.Arg2-as-detail
                      0         1             2
                -1 : [ Expansion.Level-of-detail.Arg2-as-detail ]
                 O : [ Expansion ]
                 1 : [ Level-of-detail ]
                 2 : [ Arg2-as-detail ]       
                
        '''
        relation = self._extract_relation(rel_id)
        sense_list = relation['Sense']
        sense_ouput_list = []
        if level == -1:
            return sense_list
        for sense in sense_list:
            sense_in_level = sense.strip().split('.')
            if level + 1>len(sense_in_level):
                sense_output = 'Unknown'
                print('Warning: relation {}, with sense {},  does not have level-{} sense [level: 0, 1, 2]'.format(rel_id,sense,  level))
            else:
                sense_output = sense_in_level[level]
            sense_ouput_list.append(sense_output)
        return sense_ouput_list
    
    def get_type(self, rel_id):
        relation = self._extract_relation(rel_id)
        return relation['Type']
    
    def get_sent_id(self, rel_id, Attr):
        relation = self._extract_relation(rel_id)
        doc_id = relation['DocID']
        sent_ids = list(set([ind[3] for ind in relation[Attr]['TokenList']]))
        return doc_id, [sent_id for sent_id in sent_ids]
    
    def get_rel_sent_id(self, rel_id):
        '''
        input rel_id
        return the minimum sentences id list that cover this relation
         '''
        arg1_doc_id, arg1_sents_id=self.get_sent_id(rel_id,'Arg1')
        doc_id, arg2_sents_id=self.get_sent_id(rel_id,'Arg2')
        assert doc_id == arg1_doc_id
        return doc_id, sorted(list(set(arg1_sents_id) | set(arg2_sents_id)))
    
    def get_token_id(self, rel_id, Attr):
        """
        Args:
                Attr(str): Arg1, Arg2, Connective
        Returns: 
            docid, [(sent_id, offset), ]
        """
        relation = self._extract_relation(rel_id)
        doc_id = relation['DocID']
        token_id_list = []
        for ind in relation[Attr]['TokenList']:
            if (ind[3], ind[4]) not in token_id_list:
                token_id_list.append((ind[3], ind[4]))
        return doc_id, token_id_list
    
    def __iter__(self):
        
        if (not self.iter_index_key) and (not self.iter_cond_func):  
            for rel_id in self.rel_id:
                yield rel_id
        
        elif self.iter_index_key:
            assert self.index
            if isinstance(self.iter_sub_key, list):
                assert self.iter_index_key in self.index 
                rel_id_list = []
                for sub_key in self.iter_sub_key:
                    assert sub_key in self.index[self.iter_index_key]
                    rel_id_list+= self.index[self.iter_index_key][sub_key]
                
                for rel_id in rel_id_list:
                    yield rel_id
            else:
                assert self.iter_index_key in self.index and self.iter_sub_key in self.index[self.iter_index_key]
                for rel_id in self.index[self.iter_index_key][self.iter_sub_key]:
                    yield rel_id
        
        elif self.iter_cond_func:
            for rel_id in self.rel_id:
                if self.iter_cond_func(self, rel_id):
                    yield rel_id
    
    def __call__(self, index_key=None, sub_key=None, iter_cond_func = None):
        ## FOR ITERATION
        ## you can simply access to rel_id by either key or user-defined iteration condition function
        ## example 
        ##     for rel_id in pdtb('Type','Implicit')
        ##     for rel_id in pdtb('Sense','Comparison')
        ##     for rel_id in pdtb(iter_cond_fun c)    
        ##             iter_cond_func take pdtb and rel_id as input to check whether this rel_id is qualified, user should implement the inside decision logic
        self.iter_index_key= index_key
        self.iter_sub_key = sub_key
        self.iter_cond_func = iter_cond_func
        return self
        
    def __len__(self):
        return len(rel_id)
    
class ptb3:
    def __init__(self, path, folder_list=list(range(2,24))):
        self.path = path
        self.parsing_data = self._load(folder_list)
        self.docid = self._transvere_docid()
    
    def _transvere_docid(self):
        doc_id = []
        for folder_id in self.parsing_data:
            doc_id += list(self.parsing_data[folder_id].keys())
        return doc_id
    
    def _load(self, folder_list=list(range(0,25))):
        # todo: check exist
        parsing_data_dict = {}
        for folder_ind in folder_list:
            index = '0{}'.format(folder_ind) if len(str(folder_ind)) == 1  else str(folder_ind)
            folder_name = 'wsj_{}'.format(index) 
            parsing_data_dict[folder_name] = json_load(os.path.join(self.path, folder_name +'.json'))
        return parsing_data_dict
    
    def _extract_parse_file(self, doc_id):
        folder_id = self._extract_folder_id(doc_id)
        return self.parsing_data[folder_id][doc_id]['sentences']
    
    def get_sent_num(self, doc_id):
        return len(self._extract_parse_file(doc_id))
    
    def _extract_folder_id(self, doc_id):
        return doc_id[0:6]
    
    def get_dependency(self, doc_id, token_indices):
        """
        Args:
                doc_id(str)
                token_indices[(sent#, token#), ...]
        Returns:
                dependencies[list[relation, (head(int), token(int))], ...]
        """
        
        doc = self._extract_parse_file(doc_id)
        dependencies = []
        for sent_id, token_id in token_indices:
            sent_dependency = doc[sent_id]['dependencies']
            dep_text = self._get_token_dependency(sent_dependency, token_id)
            dep_relation = dep_text[0]
            head_id = self._get_index(dep_text[1]) - 1 
            token_id = self._get_index(dep_text[2]) - 1
            dependencies.append([dep_relation, head_id, token_id])
        return dependencies
    
    def _get_token_dependency(self, sent_dependency, token_id):
        const = token_id+1
        dep_index = self._get_dep_index(token_id, sent_dependency)
        while const != dep_index:
            token_id -= (dep_index - const)
            print(token_id)
            dep_index = self._get_dep_index(token_id, sent_dependency)
        if token_id >= len(sent_dependency):
            return sent_dependency[-1]
        else:
            return sent_dependency[token_id]

    def _get_dep_index(self, index, sent_dependency):
#         return int(sent_dependency[index][2].split('-')[-1])
        if index>= len(sent_dependency): 
            assert index+1 <= self._get_index(sent_dependency[-1][2])
            return self._get_index(sent_dependency[-1][2])
        else:
            return self._get_index(sent_dependency[index][2])

    def _get_index(self, text):
        return int(text.split('-')[-1])

    def get_tokens_text(self, doc_id, token_indices):
        """
        Args:
                doc_id(str)
                token_indices[(sent#, token#), ...]
        Returns:
                token_dict{(sent#, token#): token_text, ...]
        """
        doc = self._extract_parse_file(doc_id)
        token_dict = dict()
        
        for sent_id, token_id in token_indices:
            token_text = doc[sent_id]['words'][token_id][0]
            correct_token_text = self._token_trans_(token_text)
            token_dict[(sent_id, token_id)] = correct_token_text
        return token_dict

    def get_sent_tokens_text(self, doc_id, sent_id):
        """
        Args:
                doc_id(str)
                sent_id(int) or list
        Returns:
                token_dict{(sent_number(int), token_index_in_sent(int)) : token(str), ...} 
        """
        doc = self._extract_parse_file(doc_id)
        token_dict = dict()
        if isinstance(sent_id, int):
            for i, token in enumerate(doc[sent_id]['words']):
                original_token = token[0]
                correct_token = self._token_trans_(original_token)
                token_dict[(sent_id, i)] =  correct_token
            return token_dict
        
        elif isinstance(sent_id, list):
            for sent_ind in sent_id:
                temp_dict = self.get_sent_tokens_text(doc_id, sent_ind)
                token_dict = {**token_dict, **temp_dict}
            return token_dict
        else:
            print('sent_id should be int or a list')
    
    
    def _token_trans_(self, token):
        # modify . . . -> ... since . . . will cause tokenization error in wordpiece tokenization
        special_dict = {'``': '"', '\'\'':'"', '-RRB-':')' , '-LRB-':'(', '-LCB-':'{', '-RCB-':'}', '...': '...'}
        ## original ... -> . . .
        #special_dict = {'``': '"', '\'\'':'"', '-RRB-':')' , '-LRB-':'(', '-LCB-':'{', '-RCB-':'}', '...': '. . .'}
        if token in special_dict:
            token = special_dict[token]
        if  '\\/' in token:
            token = token.replace('\\/','/')
        if '``' in token:
            token = token.replace('``','"')
        return token
        
    def get_parse_tree(self, doc_id, sent_id):
        """
        Args:
                doc_id(str)
                sentid(int)
        Returns:
                parse_tree(str)
        """
        doc = self._extract_parse_file(doc_idS)
        return Tree.fromstring(doc[sentid]['parsetree'][1:-3])
    
    def get_sent_dependency(self, doc_id, sent_id):
        doc = self._extract_parse_file(doc_id)
        dependencies = []
        # There are some sentences that DONT have dependency and constintuency trees. We have to identify it and output nothing
        if len(doc[sentid]['dependencies'])== 1 and  'ROOT' in doc[sentid]['dependencies'][0][2]:
            print(doc[sentid]['dependencies'])
            return dependencies
        
        for token_dep in doc[sentid]['dependencies']:
            dep_text = token_dep
            dep_relation = dep_text[0]
            head_id = self._get_index(dep_text[1]) - 1 
            token_id = self._get_index(dep_text[2]) - 1
            dependencies.append([dep_relation, head_id, token_id])
        return dependencies
    
    def get_sent_dependency_graph(self, doc_id, sentid):
        d_graph = Dep_Graph()
        dep_edges = self.get_sent_dependency(doc_id, sentid)
        if dep_edges:
            d_graph.build_from_dep_edge(dep_edges)
            return d_graph
        else:
            return None
    
    def __len__(self):
        return len(self.docid)
    
    def __iter__(self):
        return iter(self.docid)
