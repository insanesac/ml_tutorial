import heapq

def token_frequency(tokens):
    
    count = {}
    
    for token in tokens:
        count[token] = count.get(token, 0) + 1
        
    return count

def first_duplicate_token(tokens):
    seen = set()
    
    for token in tokens:
        if token not in seen:
            seen.add(token)

        else:
            return token

    return None

def top_k(tokens, k):
    token_count = token_frequency(tokens)
    
    if len(token_count) < 1:
        return None
    
    
    heap = [] #this is am not sire is it [] or {}
    
    for token, count in token_count.items():
        heapq.heappush(heap,(count, token))
        
        if len(heap) > k:
            heapq.heappop(heap)
            
    result = []

    while heap:
        count, token = heapq.heappop(heap)
        result.append(token)
        
    # top_k =  sorted(count, key=lambda x : x[1], reverse=True)
    # return top_k[:k]
    return result[::-1]
        
tokens = [
    "cat",
    "dog",
    "cat",
    "bird",
    "cat",
    "dog"
]

print(first_duplicate_token(tokens))

print(top_k(tokens, 2))

import math
def cosine_similarity(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot/(norm_a*norm_b)
    
a = [1,2,3]
b = [4,5,6]

print(cosine_similarity(a,b))



def longest_substring(string):
    left = 0
       
    seen = set()
    longest = 0
    for right in range(len(string)):
        while string[right] in seen:
            seen.remove(string[left])            
            left+=1
            
        seen.add(string[right])
            
        longest = max(longest, len(seen))
    return longest


string = "abcabcbb"

print(longest_substring(string))
        