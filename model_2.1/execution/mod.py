import torch
from decodeur import *
import modele_streaming as ms


class modele:
    def __init__(self):
        print ("debut")  
        prop = ""      
        while prop != "q" :
            prop = input("prompt (appuiller sur q pour quitter):  ")
            self.generation(prop, "vocab.json")
        pass
        
    
    def charger_modele():
        emb_layer, lstm, fc, optimizer, criterion = ms.init_training_objects(vocab_size=100000)
        ms.load_checkpoint_if_compatible("model_V2.1.pt", emb_layer, lstm, fc )
        return emb_layer, lstm, fc

    def gen(emb_layer, lstm, fc, tensor):
        emb = ms.embedding(tensor ,emb_layer)
        out, hidden = lstm(emb)
        last = out[:, -1,:]
        fcs = fc(last)
        gen = torch.argmax(fcs,dim=1)
        return gen

    def deocode_tot(gene, jsons):
        prid = gene[0].item()
        prid = str(prid)
        mot = decode(prid, jsons)
        return mot
    
    

    def prompt(pp):
        VOCAB = "vocab.json"
        pr = encode(pp, VOCAB)
        pr = [i % 100000 for i in pr]
        if len(pr) < 128:
            print("prompt trops court on complète")
            nb_z = 128 - len(pr)
            pr = [0] * int(nb_z) + pr
        else: 
            pass
        truncated_prompt_ids = pr[-128:]
        batch_prompt_ids = [truncated_prompt_ids]
        tens = torch.tensor(batch_prompt_ids, dtype=torch.long)
        return truncated_prompt_ids, tens


        
    def generation (self, promp, jsons):
        emb_layer, lstm, fc = modele.charger_modele()
        generation_txt = True
        x = 0
        gener = []
        context_id, tens = modele.prompt(promp)
        print("generation...")
        while generation_txt == True:           
            gen = modele.gen(emb_layer, lstm, fc, tens)
            gen = gen[0].item()
            x += 1
            gener.append(str(gen))
            context_id.append(gen)
            context_id = context_id[-128:]
            batch_prompt_ids = [context_id]
            tens = torch.tensor(batch_prompt_ids, dtype=torch.long)
            if x >= 128 :
                print(decode(" ".join(gener), jsons))
                break


if __name__ == "__main__":
    modele()
