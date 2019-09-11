python imdir_to_lmdb.py %1/images %2 %3 "--optical_flow" "--json"
python generate_pose_sequence_labels.py %1 %4 %5 "--json"
python generate_pose_sequence_key_file.py %1/images/pose_sequence_label_lmdb %1/goodkeys.txt