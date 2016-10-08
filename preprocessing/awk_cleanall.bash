cd ./CrisisLexT6/

for i in `ls`
do
echo "current dir i", $i
pwd

cd "$i"

grep "on-topic\|tweet id, tweet, label" *-ontopic_offtopic.csv > "$i"_1_temp.txt
grep "off-topic\|tweet id, tweet, label" *-ontopic_offtopic.csv > "$i"_0_temp.txt


python ../../preprocessing/python_csv.py "$i"_1_temp.txt > "$i"_1.txt
python ../../preprocessing/python_csv.py "$i"_0_temp.txt > "$i"_0.txt

../../preprocessing/awk_html_symbol_entities.pl "$i"_1.txt | awk '$0 !~/[^ -~]/' | sed 's/[^ ]*[^ ]@[^ ][^ ]*/''/g' | sed 's/@[^ ][^ ]*//g' | sed 's/http[^ ][^ ]*//gI' | sed 's/ http//gI' | sed 's/\<\(RT\)\+\>//g' | sed 's/^"*//' | sed 's/"*$//'| sed 's/^ *//' | sed 's/ *$//' | sed 's/,/ /g' | awk '$0 ~/[a-zA-Z]/ {print $0}' | sort | uniq > "$i"_1_final.txt

../../preprocessing/awk_html_symbol_entities.pl "$i"_0.txt | awk '$0 !~/[^ -~]/' | sed 's/[^ ]*[^ ]@[^ ][^ ]*/''/g' | sed 's/@[^ ][^ ]*//g' | sed 's/http[^ ][^ ]*//gI' | sed 's/ http//gI' | sed 's/\<\(RT\)\+\>//g' | sed 's/^"*//' | sed 's/"*$//'| sed 's/^ *//' | sed 's/ *$//' | sed 's/,/ /g' | awk '$0 ~/[a-zA-Z]/ {print $0}' | sort | uniq > "$i"_0_final.txt

#../../preprocessing/awk_html_symbol_entities.pl "$i"_1.txt | awk '$0 !~/[^ -~]/' | sed 's/[^ ]*[^ ]@[^ ][^ ]*/''/g' | sed 's/@[^ ][^ ]*//g' | sed 's/http[^ ][^ ]*//gI' | sed 's/ http//gI' | sed 's/ RT//gI' | sed 's/^"*//' | sed 's/"*$//'| sed 's/^ *//' | sed 's/ *$//' | sed 's/,/ /g' | awk '$0 ~/[a-zA-Z]/ {print $0}' | sort | uniq > "$i"_1_final.txt
#
#../../preprocessing/awk_html_symbol_entities.pl "$i"_0.txt | awk '$0 !~/[^ -~]/' | sed 's/[^ ]*[^ ]@[^ ][^ ]*/''/g' | sed 's/@[^ ][^ ]*//g' | sed 's/http[^ ][^ ]*//gI' | sed 's/ http//gI' | sed 's/ RT//gI' |sed 's/^"*//' | sed 's/"*$//'| sed 's/^ *//' | sed 's/ *$//' | sed 's/,/ /g' | awk '$0 ~/[a-zA-Z]/ {print $0}' | sort | uniq > "$i"_0_final.txt
#




mkdir "$i"_directory
cd "$i"_directory

mkdir 1
cd 1
awk '{ printf "%s", $0 >> NR".txt"}'  ../../"$i"_1_final.txt
cd ..

mkdir 0
cd 0
awk '{ printf "%s", $0 >> NR".txt" }'  ../../"$i"_0_final.txt
cd ..

cd ..

cd ..

done
cd ..