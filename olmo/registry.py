download_path = "/n/holyscratch01/kempner_fellows/Users/dbrandfonbrener/color-filter"
# download_path = "YOUR_PATH_HERE"

DATA_DICT = {
    "c4": f"{download_path}/full_data/c4",
    "books-small": f"{download_path}/downstream_data/books",
    "books-val": f"{download_path}/downstream_data/books_val",
}

# Pretrained model weights
MODEL_DICT = {
    "prior": f"{download_path}/models/prior",
    "conditional_books": f"{download_path}/models/conditional_books",
    "conditional_all": f"{download_path}/models/conditional_all",
    "random_1b": f"{download_path}/models/random_1b",
    "books_tau=64_1b": f"{download_path}/models/books_tau=64_1b",
    "all_tau=64_1b": f"{download_path}/models/all_tau=64_1b",
}

# Datasets of scores from auxiliary models on c4
SCORE_DICT = {
    "pretrain-1-seq": f"{download_path}/scores/pretrain-1-seq",
    "pretrain-2-seq": f"{download_path}/scores/pretrain-2-seq",
    "pretrain-3-seq": f"{download_path}/scores/pretrain-3-seq",
    "pretrain-4-seq": f"{download_path}/scores/pretrain-4-seq",
    "pretrain-5-seq": f"{download_path}/scores/pretrain-5-seq",
    "pretrain-6-seq": f"{download_path}/scores/pretrain-6-seq",
    "pretrain-7-seq": f"{download_path}/scores/pretrain-7-seq",
    "books-1-seq": f"{download_path}/scores/books-1-seq",
    "books-2-seq": f"{download_path}/scores/books-2-seq",
    "books-3-seq": f"{download_path}/scores/books-3-seq",
    "books-4-seq": f"{download_path}/scores/books-4-seq",
    "books-5-seq": f"{download_path}/scores/books-5-seq",
    "books-6-seq": f"{download_path}/scores/books-6-seq",
    "books-7-seq": f"{download_path}/scores/books-7-seq",
    "all-1-seq": f"{download_path}/scores/all-1-seq",
    "all-2-seq": f"{download_path}/scores/all-2-seq",
    "all-3-seq": f"{download_path}/scores/all-3-seq",
    "all-4-seq": f"{download_path}/scores/all-4-seq",
    "all-5-seq": f"{download_path}/scores/all-5-seq",
    "all-6-seq": f"{download_path}/scores/all-6-seq",
    "all-7-seq": f"{download_path}/scores/all-7-seq",
}

# Index files with the selected indices for each dataset
INDEX_DICT = {
    "c4-down-tau=7": f"{download_path}/indices/c4-down-tau=7/selected_indices.npy",
    "c4-down-tau=16": f"{download_path}/indices/c4-down-tau=16/selected_indices.npy",
    "c4-down-tau=32": f"{download_path}/indices/c4-down-tau=32/selected_indices.npy",
    "c4-down-tau=64": f"{download_path}/indices/c4-down-tau=64/selected_indices.npy",
    "c4-books-tau=7": f"{download_path}/indices/c4-books-tau=7/selected_indices.npy",
    "c4-books-tau=16": f"{download_path}/indices/c4-books-tau=16/selected_indices.npy",
    "c4-books-tau=32": f"{download_path}/indices/c4-books-tau=32/selected_indices.npy",
    "c4-books-tau=64": f"{download_path}/indices/c4-books-tau=64/selected_indices.npy",
}
