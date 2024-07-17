import asyncio
import websockets

connected_clients = set()

async def hello():
    while True:
        await asyncio.sleep(3)
        message = "Hello from the remote server"
        if connected_clients:
            await asyncio.wait([client.send(message) for client in connected_clients])

async def manager(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        connected_clients.remove(websocket)

start_server = websockets.serve(manager, '', 40000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().create_task(hello())
asyncio.get_event_loop().run_forever()
