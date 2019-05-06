import json
f = open('rcv1.topics.hier.orig.txt','r')
nodes = []
lines = f.readlines()
for line in lines:
    keys =line.split(' ')
    while '' in keys:
        keys.remove("")
    node=[]
    node.append(keys[1])
    node.append(keys[3])
    if(node[0]=='Root' or node[1]=='Root'):
        pass
    else:
        nodes.append(node)
f.close()

relationship = []
with open('classes.json','r') as f:
    classes = json.load(f)
    for node in nodes:
        relation = []
        relation.append(classes[node[0]])
        relation.append(classes[node[1]])
        relationship.append(relation)


with open('edgelist.txt','w') as f:
    for r in relationship:
        f.write("%d %d\n"%(r[0],r[1]))
