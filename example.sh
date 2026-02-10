python t2i/main.py --prompt "A 2-layer Refrigerator with inner drawers. The door of the refrigerator is open. Background: white." --output outputs/gen/fridge/img.png
python i2m/main.py --image outputs/gen/fridge/img.png --output outputs/gen/fridge/mesh.ply
python m2p/main.py --mesh outputs/gen/fridge/mesh.ply --output outputs/gen/fridge/parts

python t2i/main.py --prompt "A Desk with drawers and cabinet doors. The doors are open and the drawers are pulled. Background: white." --output outputs/gen/desk/img.png
python i2m/main.py --image outputs/gen/desk/img.png --output outputs/gen/desk/mesh.ply
python m2p/main.py --mesh outputs/gen/desk/mesh.ply --output outputs/gen/desk/parts

python t2i/main.py --prompt "A Washing machine with a round door. The door is open. Background: white." --output outputs/gen/washing/img.png
python i2m/main.py --image outputs/gen/washing/img.png --output outputs/gen/washing/mesh.ply
python m2p/main.py --mesh outputs/gen/washing/mesh.ply --output outputs/gen/washing/parts

python t2i/main.py --prompt "A microwave oven with a door. The door is open. Background: white." --output outputs/gen/microwave/img.png
python i2m/main.py --image outputs/gen/microwave/img.png --output outputs/gen/microwave/mesh.ply
python m2p/main.py --mesh outputs/gen/microwave/mesh.ply --output outputs/gen/microwave/parts

python t2i/main.py --prompt "A toolbox with a lid. The lid is open. Background: white." --output outputs/gen/toolbox/img.png
python i2m/main.py --image outputs/gen/toolbox/img.png --output outputs/gen/toolbox/mesh.ply
python m2p/main.py --mesh outputs/gen/toolbox/mesh.ply --output outputs/gen/toolbox/parts

python t2i/main.py --prompt "A stationery box with a lid. The lid is open. Background: white." --output outputs/gen/stationery/img.png
python i2m/main.py --image outputs/gen/stationery/img.png --output outputs/gen/stationery/mesh.ply
python m2p/main.py --mesh outputs/gen/stationery/mesh.ply --output outputs/gen/stationery/parts

python t2i/main.py --prompt "A door with handles. The door is open. Background: white." --output outputs/gen/door/img.png
python i2m/main.py --image outputs/gen/door/img.png --output outputs/gen/door/mesh.ply
python m2p/main.py --mesh outputs/gen/door/mesh.ply --output outputs/gen/door/parts

python t2i/main.py --prompt "A faucet with a handle. Background: white." --output outputs/gen/faucet/img.png
python i2m/main.py --image outputs/gen/faucet/img.png --output outputs/gen/faucet/mesh.ply
python m2p/main.py --mesh outputs/gen/faucet/mesh.ply --output outputs/gen/faucet/parts

python t2i/main.py --prompt "A delivery box. The box is open. Background: white." --output outputs/gen/delivery/img.png
python i2m/main.py --image outputs/gen/delivery/img.png --output outputs/gen/delivery/mesh.ply
python m2p/main.py --mesh outputs/gen/delivery/mesh.ply --output outputs/gen/delivery/parts

python t2i/main.py --prompt "A Flip-top trash can. The can is open. Background: white." --output outputs/gen/trash/img.png
python i2m/main.py --image outputs/gen/trash/img.png --output outputs/gen/trash/mesh.ply
python m2p/main.py --mesh outputs/gen/trash/mesh.ply --output outputs/gen/trash/parts

python m2p/main.py --mesh data/delivery.glb --output outputs/obja/delivery
python m2p/main.py --mesh data/desk.glb --output outputs/obja/desk
python m2p/main.py --mesh data/door.glb --output outputs/obja/door
python m2p/main.py --mesh data/faucet.glb --output outputs/obja/faucet
python m2p/main.py --mesh data/fridge.glb --output outputs/obja/fridge
python m2p/main.py --mesh data/microwave.glb --output outputs/obja/microwave
python m2p/main.py --mesh data/toolbox.glb --output outputs/obja/toolbox
python m2p/main.py --mesh data/trash.glb --output outputs/obja/trash
python m2p/main.py --mesh data/washing.glb --output outputs/obja/washing
