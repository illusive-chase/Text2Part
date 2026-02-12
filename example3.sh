lst=(
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
)

for idx in ${lst[@]}
do
python t2i/main.py --prompt "A realistic desk with two storage. The left part is a 3-layer drawers. The right part is not a drawer but a door-based cabinet. The doors are open and the drawers are pulled. The perspective is slightly tilted and from above, allowing most of the object's details to be seen without significant obstruction. Background: pure white." --output outputs/gen3/desk$idx/img.png --seed $idx --cfg 1.5
done
