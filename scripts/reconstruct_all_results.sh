
#stats_table
python  ~/research/github_autobi/scripts/reconstruct_metanome_results.py --keys posts_Id,users_Id,comments_Id,tags_WikiPostId  Normalize-1.2-SNAPSHOT.jar2021-02-12T191007_stats 
python  ~/research/github_autobi/scripts/reconstruct_metanome_results.py --keys posts_Id,users_Id,comments_Id,badges_Id,votes_Id  stats_badges_votes_tags_table
python  ~/research/github_autobi/scripts/reconstruct_metanome_results.py --keys "pripady_Identifikace_pripadu,pripady_Identifikace_pripadu':'vykony_Datum_provedeni_vykonu':'zup_Kod_polozky,pripady_Identifikace_pripadu':'vykony_Datum_provedeni_vykonu':'vykony_Kod_polozky"  FNHK_table
