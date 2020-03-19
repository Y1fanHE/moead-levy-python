# Usage
# python MOEAD.py [parameter yml file] [random seed]
 
for i in {1000..1010} ; do
    python MOEAD.py config.yml ${i}
done
