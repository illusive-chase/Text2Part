# python t2i/main.py --prompt "A 2-layer Refrigerator with inner drawers. The door of the refrigerator is open. Background: white." --output outputs/fridge/img.png
# python i2m/main.py --image outputs/fridge/img.png --output outputs/fridge/mesh.ply
# python m2p/main.py --mesh outputs/fridge/mesh.ply --output outputs/fridge/parts

# python t2i/main.py --prompt "A Desk with drawers and cabinet doors. The doors are open and the drawers are pulled. Background: white." --output outputs/desk/img.png
# python i2m/main.py --image outputs/desk/img.png --output outputs/desk/mesh.ply
# python m2p/main.py --mesh outputs/desk/mesh.ply --output outputs/desk/parts

python t2i/main.py --prompt "A Washing machine with a round door. The door is open. Background: white." --output outputs/washing/img.png
python i2m/main.py --image outputs/washing/img.png --output outputs/washing/mesh.ply
python m2p/main.py --mesh outputs/washing/mesh.ply --output outputs/washing/parts

python t2i/main.py --prompt "A microwave oven with a door. The door is open. Background: white." --output outputs/microwave/img.png
python i2m/main.py --image outputs/microwave/img.png --output outputs/microwave/mesh.ply
python m2p/main.py --mesh outputs/microwave/mesh.ply --output outputs/microwave/parts

python t2i/main.py --prompt "A toolbox with a lid. The lid is open. Background: white." --output outputs/toolbox/img.png
python i2m/main.py --image outputs/toolbox/img.png --output outputs/toolbox/mesh.ply
python m2p/main.py --mesh outputs/toolbox/mesh.ply --output outputs/toolbox/parts

python t2i/main.py --prompt "A stationery box with a lid. The lid is open. Background: white." --output outputs/stationery/img.png
python i2m/main.py --image outputs/stationery/img.png --output outputs/stationery/mesh.ply
python m2p/main.py --mesh outputs/stationery/mesh.ply --output outputs/stationery/parts

python t2i/main.py --prompt "A door with handles. The door is open. Background: white." --output outputs/door/img.png
python i2m/main.py --image outputs/door/img.png --output outputs/door/mesh.ply
python m2p/main.py --mesh outputs/door/mesh.ply --output outputs/door/parts

python t2i/main.py --prompt "A faucet with a handle. Background: white." --output outputs/faucet/img.png
python i2m/main.py --image outputs/faucet/img.png --output outputs/faucet/mesh.ply
python m2p/main.py --mesh outputs/faucet/mesh.ply --output outputs/faucet/parts

python t2i/main.py --prompt "A delivery box. The box is open. Background: white." --output outputs/delivery/img.png
python i2m/main.py --image outputs/delivery/img.png --output outputs/delivery/mesh.ply
python m2p/main.py --mesh outputs/delivery/mesh.ply --output outputs/delivery/parts

python t2i/main.py --prompt "A Flip-top trash can. The can is open. Background: white." --output outputs/trash/img.png
python i2m/main.py --image outputs/trash/img.png --output outputs/trash/mesh.ply
python m2p/main.py --mesh outputs/trash/mesh.ply --output outputs/trash/parts

