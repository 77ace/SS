import numpy as np
def AdaptiveSampling(alternatives, similarity):
    """
    Adaptive sampling function to select a word based on its similarity score.
    """
    #print("-------Adaptive Sampling-------")
    # Normalize similarity scores to probabilities
    total_similarity = sum(similarity)
    #print(f"-------totalSim= {total_similarity}-------")

    #calculate probabilities of each alternative, all probabilities should sum up to 1
    probabilities=[]
    for i in range(len(similarity)):
        probabilities.append(similarity[i] / total_similarity)
    #print(f"-------propabilites= {probabilities}-------")

    # Select a word based on the computed probabilities
    selected_word = np.random.choice(alternatives, p=probabilities)
    #print(f"-------selected_word= '{selected_word}'-------")

    return selected_word

import hashlib
#function to get a hashed binary value based on a target, candidate, and threshold,
#this fuction is used by the HashBasedSampling function
def getHasedBinaryValue(target, candidate, threshold, key ):
    """
    get hashed binary value (0 or 1) for a given key, target and candidate
    """
    hash_value = hashlib.sha256(f"{key}_{target}_{candidate}".encode('utf-8')).hexdigest()
    # Convert the hash value to an integer
    hash_int = int(hash_value, 16)
    # Normalize the hash value to a float between 0 and 1
    normalized_value = hash_int / (2**256 - 1)
    # Return 1 if the normalized value is greater than the threshold, else return 0
    if normalized_value > threshold:
        return 1
    else:
        return 0
    
# Hash-based sampling function to select a word based on its similarity score
def HashBasedSampling(alternatives, similarity ,target ,hash_key, threshold=0.5, ):
    """
    Hash-based sampling function to select a word based on its similarity score.
    """
    binary_values = []
    for alt in alternatives:
        # Calculate the binary value for each alternative
        binary_value = getHasedBinaryValue( target=target , candidate=alt, threshold=threshold, key=hash_key)
        binary_values.append(binary_value)
    print(f"-------binary_values= {binary_values}-------")


    #check the words that have a binary value of 1
    candidates = []
    for i, value in enumerate(binary_values):
        if value == 1:
            candidates.append((alternatives[i], similarity[i]))
    
    print(f"-------candidates= {candidates}-------")

    # If no candidates are found (binary values all 0), return original word
    if not candidates:
        print("No candidates found based on the hash-based sampling.")
        return target
    elif len(candidates) == 1:
        # If only one candidate is found, return it
        selected_word = candidates[0][0]
        print(f"-------selected_word= '{selected_word}'-------")
        return selected_word
    else:
        #get the candidate with the highest similarity score
        selected_word = max(candidates, key=lambda x: x[1])[0]
        print(f"-------selected_word= '{selected_word}'-------")
        return selected_word
    


# #test the HashBasedSampling function
# alternatives = ["stunning", "pretty", "gorgeous"]
# similarities = [0.95, 0.94, 0.80]

# HashBasedSampling(alternatives, similarities, "beautiful")
