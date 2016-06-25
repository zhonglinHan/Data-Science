## R version of most popular local hotels
library(data.table)
expedia_train <- fread('../input/train.csv', header=TRUE, select= c("is_booking","orig_destination_distance","hotel_cluster","srch_destination_id"))
expedia_test <- fread('../input/test.csv', header=TRUE)

sum_and_count <- function(x){
  sum(x)*0.8456 + length(x) *(1-0.8456)
}

dest_id_hotel_cluster_count <- expedia_train[,sum_and_count(is_booking),by=list(orig_destination_distance, hotel_cluster)]
dest_id_hotel_cluster_count1 <- expedia_train[,sum_and_count(is_booking),by=list(srch_destination_id, hotel_cluster)]


top_five <- function(hc,v1){
  hc_sorted <- hc[order(v1,decreasing=TRUE)]
  n <- min(5,length(hc_sorted))
  paste(hc_sorted[1:n],collapse=" ")
}

dest_top_five <- dest_id_hotel_cluster_count[,top_five(hotel_cluster,V1),by=orig_destination_distance]
dest_top_five1 <- dest_id_hotel_cluster_count1[,top_five(hotel_cluster,V1),by=srch_destination_id]

dd <- merge(expedia_test,dest_top_five, by="orig_destination_distance",all.x=TRUE)[order(id),list(id,V1)]

dd1 <- merge(expedia_test,dest_top_five1, by="srch_destination_id",all.x=TRUE)[order(id),list(id,V1)]

dd$V1[is.na(dd$V1)] <- dd1$V1[is.na(dd$V1)] 

setnames(dd,c("id","hotel_cluster"))

write.csv(dd, file='submission_combo_merge.csv', row.names=FALSE)
