
import heapq

def count_tokens(tokens):
    
    token_count = {}    
    
    for token in tokens:
        token_count[token] = token_count.get(token,0) + 1
        
    #sorted
    #sorted_counts = sorted(toekn_count.items(), key = lambda x : x[1], reverse=True)
    #return sorted_counts[:k]
    
    heap = []
    k = 3
    for token, count in token_count.items():
        heapq.heappush(heap, (count, token))
         
        if len(heap) > k:
            heapq.heappop(heap)
            
    results = []
    
    while heap:
        count, token = heapq.heappop(heap)
        result.append(token)

   
    return result


def max_sum_subarray(nums, k):
    if len(nums) < k:
        return None
    
    window_sum = sum(nums[:k])
    
    max_sum = window_sum
    
    for i in range(k, len(nums)):
        print(i)
        window_sum = window_sum - nums[i-k] + nums[i]
        
nums = [1,2,3,4,5]
k = 3

max_sum_subarray(nums,k)



def two_sum_sorted(nums, target):
    left = 0
    right = len(nums)-1
    
    while left < right:
        sum = nums[left] + nums[right]
        
        if sum == target:
            return [left, right]
        
        if sum < target:
            left += 1
        else:
            right -= 1
            
    return []

nums = [1,2,4,6,8,10]
target = 12

print(two_sum_sorted(nums, target))

def first_greater(nums, target):
    left = 0
    right = len(nums) - 1
    
   
    greater = None
    while left <= right:
        mid = (left + right) // 2

        
        if target < nums[mid]:
            right = mid -1
            greater = mid
        elif target >= nums[mid]:
            left = mid + 1
            
            
    return greater 




nums = [1,3,5,7,9]
target = 4

print(first_greater(nums, target))                     


def binary_search(nums, target):
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        mid = (right + left) // 2
        if nums[mid] == target:
            return mid
        elif target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
            
            

def mrr(ranked, relevant):
    for i, r in enumerate(ranked):
        if r in relevant:
            mrr = 1 /(i+1)
            
            return mrr
        
    
ranked_docs = [10,20,30,40]
relevant_docs = {30,50}

print(mrr(ranked_docs, relevant_docs))


def recall_k(retrieved, relevant):
    if not relevant:
        return 0

    count = 0
    for r in retrieved:
        if r in relevant:
            count+=1
            
    return count/len(relevant)


def chunk_text(tokens, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError(...)
    
    output = []
    
    start = 0
    
    while start < len(tokens):
        chunk = tokens[start:start+chunk_size]
        output.append(chunk)    
        start += (chunk_size-overlap)    

    return output

tokens = list(range(20))

chunk_size = 5
overlap = 2

print(chunk_text(tokens, chunk_size, overlap))

def truncate_context(messages, token_limit):
    total_count = 0
    
    output = []
    for message, token in reversed(messages):
        print(total_count, token)
        if total_count+token > token_limit:
            break

        output.append((message, token))
        total_count+=token
    
    output.reverse()
    return output


messages = [
    ("msg1", 100),
    ("msg2", 200),
    ("msg3", 300),
    ("msg4", 400)
]

token_limit = 500

print(truncate_context(messages, token_limit))