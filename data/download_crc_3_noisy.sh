wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1QvVxUSieDiz0apIHGRUr_CjngJufq6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1QvVxUSieDiz0apIHGRUr_CjngJufq6" -O Y_train.npy && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J2kqbDdOoUfl3mOlxE9RrAMT980-IlJ5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J2kqbDdOoUfl3mOlxE9RrAMT980-IlJ5" -O Y_test.npy && rm -rf /tmp/cookies.txti
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vPT2bu6oF2RdlbVk8w8gakPwUbaoTfyT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vPT2bu6oF2RdlbVk8w8gakPwUbaoTfyT" -O X_train.npy && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Cyqvyw9uKQzmGzKeUF43FjvigE8Gy21x' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Cyqvyw9uKQzmGzKeUF43FjvigE8Gy21x" -O X_test.npy && rm -rf /tmp/cookies.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hr6v7OsP_oJbfM5gjt6sVI1rNPibw8Ts' -O real_classes_train.npy
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c0YGCgjvzPjyBVWDem_8069g-aK2VMzl' -O real_classes_test.npy
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NCnPVZILURZgQVC3_Y8TbEGipFcV4iKs' -O patients_train.npy
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wf5pcz4Jd3EbL8ckusH8BCg3GjSInQJi' -O patients_test.npy
