lst=(
Solid wood, Modern minimalist
Solid wood, Industrial
Solid wood, Nordic
Solid wood, Japanese
Solid wood, Vintage
Solid wood, French
Solid wood, Light luxury
Solid wood, Chinese style
Bamboo, Modern minimalist
Bamboo, Industrial
Bamboo, Nordic
Bamboo, Japanese
Bamboo, Vintage
Bamboo, French
Bamboo, Light luxury
Bamboo, Chinese style
Metal + wood, Modern minimalist
Metal + wood, Industrial
Metal + wood, Nordic
Metal + wood, Japanese
Metal + wood, Vintage
Metal + wood, French
Metal + wood, Light luxury
Metal + wood, Chinese style
Metal + glass, Modern minimalist
Metal + glass, Industrial
Metal + glass, Nordic
Metal + glass, Japanese
Metal + glass, Vintage
Metal + glass, French
Metal + glass, Light luxury
Metal + glass, Chinese style
Engineered wood, Modern minimalist
Engineered wood, Industrial
Engineered wood, Nordic
Engineered wood, Japanese
Engineered wood, Vintage
Engineered wood, French
Engineered wood, Light luxury
Engineered wood, Chinese style
Acrylic, Modern minimalist
Acrylic, Industrial
Acrylic, Nordic
Acrylic, Japanese
Acrylic, Vintage
Acrylic, French
Acrylic, Light luxury
Acrylic, Chinese style
Rattan + wood, Modern minimalist
Rattan + wood, Industrial
Rattan + wood, Nordic
Rattan + wood, Japanese
Rattan + wood, Vintage
Rattan + wood, French
Rattan + wood, Light luxury
Rattan + wood, Chinese style
)

for idx in ${!lst[@]}
do
python t2i/main.py --prompt "A realistic, ${lst[$idx]} desk with two storage. The left part is a 3-layer drawers. The right part is not a drawer but a door-based cabinet. The doors are open and the drawers are pulled. The perspective is slightly tilted and from above, allowing most of the object's details to be seen without significant obstruction. Background: pure white." --output outputs/gen2/desk$idx/img.png
done
