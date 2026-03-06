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

# for idx in ${lst[@]}
# do
# python t2i/main.py --prompt "Scissors. The scissors is open. The perspective is slightly tilted and from above, allowing most of the object's details to be seen without significant obstruction. Keep the entire object included in the field of view without being cropped. Background: pure white." --output outputs/gen4/scissors$idx/img.png --seed $idx --cfg 1.5 --aspect 1:1
# done

for idx in ${lst[@]}
do
# python i2m/main.py --image outputs/gen4/scissors$idx/img.png --output outputs/gen4/scissors$idx/mesh.ply
python m2p/main.py --mesh outputs/gen4/scissors$idx/mesh.ply --output outputs/gen4/scissors$idx/parts
done
