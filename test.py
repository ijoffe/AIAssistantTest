import asyncio
import websockets

connected_clients = set()

async def hello():
    while True:
        await asyncio.sleep(3)
        # message = input()
        message = "**Hazard**: [Description of the hazard]\n**Explanation**: [Brief explanation of why it is a hazard]\n**Suggestion**: [Suggestion to eliminate or reduce the danger]"
        print("Message posted")
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
