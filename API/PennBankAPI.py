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
        self.path = path
        self.relation_data = self._load(folder_list)
        self.rel_id2docidOffset = self._build_rel_id2docidOffset_map()
        self.rel_id = list(self.rel_id2docidOffset.keys())
    
    def _build_rel_id2docidOffset_map(self):
        rel_id2docidOffset = {}
        for folder_id in self.relation_data:
            for doc_id in self.relation_data[folder_id]:
                for offset, relation in enumerate(self.relation_data[folder_id][doc_id]):
                    rel_id2docidOffset[relation['ID']] = (doc_id, offset)
        return rel_id2docidOffset
    
    def query_rel_id(doc_id, offset):
        folder_id = self._extract_folder_id(self, docid)
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
        # todo: check exist
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
            docid, offset = self.rel_id2docidOffset[argv[0]]
        else:
            assert isinstance(argv[0], str)
            assert isinstance(argv[1], int)
            docid = argv[0]
            offset = argv[1]
        
        folder_id = self._extract_folder_id(docid)
        return self.relation_data[folder_id][docid][offset]
    
    def _extract_folder_id(self, docid):
        return docid[0:6]
    
    def get_raw_text(self, rel_id, Attr):
        relation = self._extract_relation(rel_id)
        return relation[Attr]['RawText']
    
    def get_connective(self, rel_id):
        relation = self._extract_relation(rel_id)
        return relation['Connective']['RawText']
    
    def get_sense(self, rel_id, level=0):
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
        docid = relation['DocID']
        sent_ids = list(set([ind[3] for ind in relation[Attr]['TokenList']]))
        return [(docid, sent_id) for sent_id in sent_ids]
    
    def get_rel_sent_id(self, rel_id):
        '''
        input rel_id
        return the minimum sentences id list that cover this relation
         '''
        docid, arg1_sents_id=zip(*self.get_sent_id(rel_id,'Arg1'))
        docid, arg2_sents_id=zip(*self.get_sent_id(rel_id,'Arg2')) 
        return docid[0], sorted(list(set(arg1_sents_id) | set(arg2_sents_id)))
    
    def get_token_id(self, rel_id, Attr):
        """
        Args:
                Attr(str): Arg1, Arg2, Connective
        Returns: list of all the sentence id containing x
        """
        relation = self._extract_relation(rel_id)
        docid = relation['DocID']
        token_id_list = []
        for ind in relation[Attr]['TokenList']:
            if (docid, (ind[3], ind[4])) not in token_id_list:
                token_id_list.append((docid, (ind[3], ind[4])))
        return token_id_list
    
    def __iter__(self):
        if not self.iter_index_key:  
            return iter(self.rel_id)
        else:
            assert self.index
            if isinstance(self.iter_sub_key, list):
                assert self.iter_index_key in self.index 
                rel_id_list = []
                for sub_key in self.iter_sub_key:
                    assert sub_key in self.index[self.iter_index_key]
                    rel_id_list+= self.index[self.iter_index_key][sub_key]
                return iter(rel_id_list)
            else:
                assert self.iter_index_key in self.index and self.iter_sub_key in self.index[self.iter_index_key]
                return iter(self.index[self.iter_index_key][self.iter_sub_key])
    
    def __call__(self, index_key=None, sub_key=None):
        self.iter_index_key= index_key
        self.iter_sub_key = sub_key
        return self
        
    def __len__(self):
        return len(rel_id)
    
class ptb3:
    def __init__(self, path, folder_list=list(range(2,24))):
        self.path = path
        self.parsing_data = self._load(folder_list)
        self.docid = self._transvere_docid()
    
    def _transvere_docid(self):
        docid = []
        for folder_id in self.parsing_data:
            docid += list(self.parsing_data[folder_id].keys())
        return docid
    def _load(self, folder_list=list(range(0,25))):
        # todo: check exist
        parsing_data_dict = {}
        for folder_ind in folder_list:
            index = '0{}'.format(folder_ind) if len(str(folder_ind)) == 1  else str(folder_ind)
            folder_name = 'wsj_{}'.format(index) 
            parsing_data_dict[folder_name] = json_load(os.path.join(self.path, folder_name +'.json'))
        return parsing_data_dict
    
    def _extract_parse_file(self, docid):
        folder_id = self._extract_folder_id(docid)
        return self.parsing_data[folder_id][docid]['sentences']
    
    def get_sent_num(self, docid):
        return len(self._extract_parse_file(docid))
    
    def _extract_folder_id(self, docid):
        return docid[0:6]
    
    def get_dependency(self, docid, token_indices):
        """
        Args:
                docid(str)
                token_indices[(sent#, token#), ...]
        Returns:
                dependencies[list[relation, (head(int), token(int))], ...]
        """
        
        doc = self._extract_parse_file(docid)
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

    def get_tokens_text(self, docid, token_indices):
        """
        Args:
                docid(str)
                token_indices[(sent#, token#), ...]
        Returns:
                token_list[token(str], ...]
        """
        doc = self._extract_parse_file(docid)
        token_list = []
        for sent_id, token_id in token_indices:
            token_text = doc[sent_id]['words'][token_id][0]
            correct_token_text = self._token_trans_(token_text)
            token_list.append(correct_token_text)
        return token_list

    def get_sent_tokens_text(self, docid, sent_id):
        """
        Args:
                docid(str)
                sent_id(int)
        Returns:
                token_list[ (sent_number(int), token_index_in_sent(int), token(str)), ...]
        """
        doc = self._extract_parse_file(docid)
        token_list = []
        for i, token in enumerate(doc[sent_id]['words']):
            original_token = token[0]
            correct_token = self._token_trans_(original_token)
            token_list.append( (sent_id, i, correct_token) )
        return token_list
    
    
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
        
    def get_parse_tree(self, docid, sentid):
        """
        Args:
                docid(str)
                sentid(int)
        Returns:
                parse_tree(str)
        """
        doc = self._extract_parse_file(docid)
        return Tree.fromstring(doc[sentid]['parsetree'][1:-3])
    
    def get_sent_dependency(self, docid, sentid):
        doc = self._extract_parse_file(docid)
        dependencies = []
        # There are some sentences without dependency and constintuency trees. we have to identify it and output nothing
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
    
    def get_sent_dependency_graph(self, docid, sentid):
        d_graph = Dep_Graph()
        dep_edges = self.get_sent_dependency(docid, sentid)
        if dep_edges:
            d_graph.build_from_dep_edge(dep_edges)
            return d_graph
        else:
            return None
    
    def __len__(self):
        return len(self.docid)
    
    def __iter__(self):
        return iter(self.docid)
