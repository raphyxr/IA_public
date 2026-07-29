import json
from itertools import islice
print("debut")

input_file = "/content/drive/MyDrive/dataset_token.txt"
output_file = "/content/drive/MyDrive/dataset-IDs.txt"
vocab_file = "/content/drive/MyDrive/vocab_l.json"


x = 0
vocab = {}


def lire_par_blocks_et_main(fichier, taille):
    global x
    global vocab
    print("lancement de la lecture")
    
    
    vocab = {}
    with open(fichier, "r", encoding="utf-8") as f : 
        
        while True:
            print (f"lire {taille} lignes de {fichier}")
            
            liste = list(islice(f, taille))

            if not liste:
                    print("fin")
                    break
            
            print("chunk traité de 1000 lignes")
            
            for lign in liste:
            
                
                
                for tok in lign.split():
                    
                    ID = None
                       
                    if tok in vocab:
                        second = vocab[tok]
                        ID = second
                        IDS_2 = [tok, second]                   
                    
                    else:
                        IDS_2 = [tok, x]       
                        vocab[tok] = x
                        ID = x 
                        x += 1        

                   
                                        

                    with open (output_file, "a", encoding="utf-8") as f_out:
                        f_out.write(str(ID) + " ")

            
                    
    
                    
  
lire_par_blocks_et_main(input_file, 1000)
print("finit de loader")


with open (vocab_file, "w", encoding="utf-8") as f_vocab:
    json.dump(vocab, f_vocab, ensure_ascii=False, indent=2)

print ("finit d'ecrire le voc")
print("FIN")






