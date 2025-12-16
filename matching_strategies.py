#argmax function to find best matching patch
def find_best_match_argmax(s, width):
    best_match_idx = s.argmax().item()#argmax over the similarities
    y = best_match_idx // width
    x = best_match_idx % width
    return x, y

#will later add more matching strategies here