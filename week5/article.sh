array[0]="US-W"
array[1]="US-c"
array[2]="HK"
array[3]="FR"
array[4]="DE"
array[5]="US"
array[6]="CA"
array[6]="CA-W"


size=${#array[@]};
index=$(($RANDOM % $size));
sudo -s /etc/init.d/windscribe-cli start ;
for i in `seq 1 5 6402`;do
index=$(($RANDOM % $size));
sudo -s /etc/init.d/windscribe-cli start ;
windscribe   connect ${array[$index]};
sleep 5;
python web_scrape_article.py $i ;
done

#for i in `seq 1 5 6403`; 
#sudo /etc/init.d/windscribe-cli start
#for i in `cat Art*`;do wget -m  http://str.sg/${i} ;done1


