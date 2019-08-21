# PDTB_API
python version: >= 3.6.8
package requirement:
- json
- nltk
- networkx

Data:
- Preprocess file
  - PTB parsed file
  - PDTB relation file (in CONLL style)
  
Differen between PennBankAPI and PennBankAPI2:
- Relation access iteration difference
  - PennBankAPI2 support user-definded iteration method, which means you can access to relation data based on your iteration rule. For example, you can select data of 'Implict Expansion' and 'Explicit Contingency' type only
  - PennBankAPI only support access to data by key and its value. For example you can only select 'Type: Implict' or 'Sense: Expansion.aa.bb', but you cannot combine these two selection criteria together. Or you have to add some extra code inside for loop
- Output format
  - Given my own coding experience, I have modified some output format to facilitate following coding.
  - (PDTB) sent_id output format: [(doc_id, sent_id), ...]  (PennBankAPI)  -> doc_id, [sent_id, ...] (PennBankAPI2)  (since a relation cannot emerge in different passages)
  - (PDTB) token_id output format: [(doc_id, (sent_id, offset)), ...]  (PennBankAPI)  -> doc_id, [(sent_id, offset), ...] (PennBankAPI2)
  - (PTB) token_text output format: [(sent_id, offset, token_text), ...] (list) -> {(sent_id, offset): token_text,  } (dict, pay attention when you use python with version lower than 3.6.8, wher dict may not be order dict by default. In that case, when you access to the token text, the dict may output the token text which does not follow original order, i.e., (1,1) -> (1,2) -> (1,3), ... . To make sure that the output follow the correct order, you may have to sort the dict key first. Then use the sorted key list to retrieve the token texts)
- Misc.
  - build index by default in PDTB initialization step, i.e., build_index(['Sense','Type'])
