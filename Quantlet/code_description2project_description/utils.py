import sys 
sys.path.append('../Create_description')

from preprocessing_utils import QuantletDataset

class QuantletDescriptionMeta(QuantletDataset):
    
    def __init__(self, parsed_Qs_file):
         super().__init__(parsed_Qs_file)

    def __getitem__(self, index):
        subset = self.parsed_Qs_file.iloc[index, :]
        code_script_path = os.path.join(subset.folder_name, subset.script_name)
        try:
            code_snippet = load_code(code_script_path, subset.type_script)
        except:
            self.dw += 1
            code_snippet = 'empty'
        return code_snippet, subset
