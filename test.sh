# python local_analysis.py -gpuid '2,3' \
#                          -modeldir 'saved_models/multihead005' \
#                          -model '50push0.8237.pth' \
#                          -imgdir 'datasets/CUB_200_2011/test/071.Long_tailed_Jaeger' \
#                          -img 'Long_Tailed_Jaeger_0046_797103.jpg' \
#                          -imgclass 70

# python local_analysis.py -gpuid '2,3' \
#                          -modeldir 'saved_models/multihead005' \
#                          -model '50push0.8237.pth' \
#                          -imgdir 'datasets/CUB_200_2011/test/035.Purple_Finch' \
#                          -img 'Purple_Finch_0046_27295.jpg' \
#                          -imgclass 34
                         
python global_analysis.py -gpuid '0, 1' \
                         -modeldir 'saved_models/tune105_onlymask&dataaug' \
                         -model '50push0.8420.pth'