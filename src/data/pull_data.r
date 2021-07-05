library(sbtools)

##################################################################################
# (Jared - Sept 2020) - pull all data needed for MTL paper from sciencebase
# (note) - in the future should use queries if available instead of hard coding IDs
#################################################################################
cat("Enter ScienceBase username: ");
un <- readLines("stdin",n=1);
cat("Enter ScienceBase password: ");
pas <- readLines("stdin",n=1);
cat( "\n" )

authenticate_sb(un,pas)


dest_dir = '../../data/raw/data_release/'


#pb0 predictions
item_file_download('5ebe569582ce476925e44b2f',overwrite_file=TRUE,dest_dir=dest_dir)


# model inputs 
item_file_download('5ebe568182ce476925e44b2d',overwrite_file=TRUE,dest_dir=dest_dir)

#lake metadata
item_file_download('5ebe564782ce476925e44b26?f=__disk__59%2Fe4%2Fac%2F59e4ac7164496cf60ad5db619349c9caf93e8152',overwrite_file=TRUE,dest_dir=dest_dir)


#temperature obs (updated 06/25/21)
item_file_download('60341c3ed34eb12031172aa6?f=__disk__64%2Fba%2Fa3%2F64baa3d97c8c89eed62b3c162ddd04ea770a4cc9',overwrite_file=TRUE,dest_dir=dest_dir)

# pb0 configs
item_file_download('5ebe567782ce476925e44b2b?f=__disk__87%2F71%2F17%2F877117acec395850457dfa317c0d576b453bcdda',overwrite_file=TRUE,dest_dir=dest_dir)


