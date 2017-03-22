import random

#Represents a graph, with edgea[v] listing the vertices adjacent to v
class graph:
	n=0
	edges=[]
	meanDegree=0


#Finds the next integer in a string after start and returns it
def nextInt(s,start):
	if start>=len(s):
		return None
	i=start
	while (s[i]<'0' or s[i]>'9'):
		i=i+1
		if i>=len(s):
			return None
	j=i
	while(s[j]>='0' and s[j]<='9' and j<len(s)):
		j=j+1
	return (int(s[i:j]),j)


#Determines how many vertices in a graph have degrees in a given range.
def degreeRange(minDegree,maxDegree,Gr):
	count=0
	for i in Gr.edges:
		if len(i)>=minDegree and len(i)<=maxDegree:
			count=count+1
	return count


#Reads a graph from the given file.
#This was written for the formatting of polblogs.gml, so it may not work on other files.
#fileName should give the full path to the file.
def readGraph1(fileName):
    dataFile=open(fileName,'r')
    data=dataFile.read()
    maxId=0
    for i in range(len(data)):
        if data[i:i+4]==' id ':
                num=nextInt(data,i)[0]
                if num>maxId:
                        maxId=num
    G=graph()
    G.n=maxId
    G.edges=[[] for i in range(G.n)]
    for i in range(len(data)):
        if data[i:i+6]==' edge ':
                (start,j)=nextInt(data,i)
                end=nextInt(data,j)[0]-1
                start=start-1
                if(not (end in G.edges[start])):
                        G.edges[start].append(end)
                if(not (start in G.edges[end])):
                        G.edges[end].append(start)                   
    degreeSum=sum([len(i) for i in G.edges])
    G.meanDegree=degreeSum/G.n
    return G			



#Finds all vertices sufficinetly close to a given vertex, where the distance is set to the minimum for which it finds at least s vertices.
#balls[i] is a lsit of all vertices at distance i, while distances[v2] gives the distance from v to v2 if it is within range.
def findNeighborhood(Gr, v, s):
	balls=[[v]]
	distances={}
	distances[v]=0
	s1=1
	d=0
	list1=[v]
	while(s1<s):
		d=d+1
		list2=[]
		for v2 in list1:
			for v3 in Gr.edges[v2]:
				if not v3 in distances.keys():
					distances[v3]=d
					list2.append(v3)
		list1=list2
		balls.append(list1)
		s1=s1+len(list1)
		if len(list1)==0:
			return (balls,distances,d)
	return (balls,distances,d)


#Determines the fraction of pairs of an edge leaving the given balls that are distinct but hit the same vertex. 
def checkOverlap(Gr,x,y,d1,d2):
	count=0
	edgeCount1=sum([len(Gr.edges[v])-1 for v in y[0][d2]])
	edgeCount2=sum([len(Gr.edges[v])-1 for v in x[0][d1]])
	for v in y[0][d2]:
		for v2 in Gr.edges[v]:
			if ((not v2 in y[1].keys()) or y[1][v2]>d2) and ((not v2 in x[1].keys()) or x[1][v2]>d1):
				for v3 in Gr.edges[v2]:
					if(v3 in x[1].keys() and x[1][v3]==d1 and v3!=v):
						count=count+1
	if(edgeCount1*edgeCount2==0):
		return None
	return count/edgeCount1/edgeCount2



#Attempts to determine how similar two vertices are by checking the fraction of pairs of edges leaving the balls of given radii centered on them hit the same vertex without beign the same.
#Using s=8 in findNeigborhood is ad hoc and might need to be changed for other graphs.
def compareVert(Gr,v1,v2):
	if( len(Gr.edges[v1])*len(Gr.edges[v2])==0):
	    return None
	x=findNeighborhood(Gr,v1,8)
	y=findNeighborhood(Gr,v2,8)
	return checkOverlap(Gr,x,y,x[2],y[2])



#Divides a graph's vertices into 2 communities.
def classify(Gr):
	compSum=0
	count=0
	while count<30:
		v1=random.randint(0,Gr.n-1)
		v2=random.randint(0,Gr.n-1)
		x=compareVert(Gr,v1,v2)
		if (x!=None):
			compSum=compSum+x
			count=count+1
	compMean=compSum/count
	refFound=False
	while(not refFound):
		r1=random.randint(0,Gr.n-1)
		r2=random.randint(0,Gr.n-1)
		x=compareVert(Gr,r1,r2)
		refFound=(x!=None and x<compMean and len(Gr.edges[r1])>Gr.meanDegree and len(Gr.edges[r2])>Gr.meanDegree)
	output=[]
	for i in range(Gr.n):
		x=compareVert(Gr,r1,i)
		y=compareVert(Gr,r2,i)
		if(x==None or y==None):
			output.append(None)
		else:
			if(x<y):
				output.append(0)
			else:
				output.append(1)
	return output


#Runs classify 3 times and refines these classification using the classifications of the vertices' neighbors.
#Then it assigns each vertex a community based on a majority vote.
def classify2(Gr):
        classes=[classify(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=random.randint(0,1)
                        if classes[-1][i]==None:
                                newClasses[i]=None
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c1=newClasses
        classes=[classify(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=random.randint(0,1)
                        if classes[-1][i]==None:
                                newClasses[i]=None
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c2=newClasses
        classes=[classify(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=random.randint(0,1)
                        if classes[-1][i]==None:
                                newClasses[i]=None
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c3=newClasses
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c1[i]!=None and c2[i]!=1 and c1[i]==c2[i]:
                        matchCount=matchCount+1
                if c1[i]!=None and c2[i]!=1 and c1[i]+c2[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r12=1
        if(matchCount<=disCount):
                r12=-1
        rs12=max(matchCount,disCount)
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c1[i]!=None and c3[i]!=1 and c1[i]==c3[i]:
                        matchCount=matchCount+1
                if c1[i]!=None and c3[i]!=1 and c1[i]+c3[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r13=1
        if(matchCount<=disCount):
                r13=-1
        rs13=max(matchCount,disCount)
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c2[i]!=None and c3[i]!=1 and c2[i]==c3[i]:
                        matchCount=matchCount+1
                if c2[i]!=None and c3[i]!=1 and c2[i]+c3[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r23=1
        if(matchCount<=disCount):
                r23=-1
        rs23=max(matchCount,disCount)
        if(r12*r13*r23==-1):
                if(rs12<rs13 and rs12<rs23):
                        r12=-r12
                else:
                        if(rs13<rs23):
                                r13=-r13
                        else:
                                r23=-r23
        output=[None]*Gr.n			
        for i in range(Gr.n):
                if(c1[i]!=None and c2[i]!=None and c3[i]!=None):
                        balance=c1[i]+r12*c2[i]+r13*c3[i]
                        if balance>0:
                                output[i]=1
                        else:
                                output[i]=0
        return output


#Runs classify2 3 times and refines these classification using the classifications of the vertices' neighbors.
#Then it assigns each vertex a community based on a majority vote.
def classify3(Gr):
        classes=[classify2(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=classes[-1][i]
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c1=newClasses
        classes=[classify2(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=classes[-1][i]
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c2=newClasses
        classes=[classify2(Gr)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in Gr.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=classes[-1][i]
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)
        c3=newClasses
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c1[i]!=None and c2[i]!=1 and c1[i]==c2[i]:
                        matchCount=matchCount+1
                if c1[i]!=None and c2[i]!=1 and c1[i]+c2[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r12=1
        if(matchCount<=disCount):
                r12=-1
        rs12=max(matchCount,disCount)
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c1[i]!=None and c3[i]!=1 and c1[i]==c3[i]:
                        matchCount=matchCount+1
                if c1[i]!=None and c3[i]!=1 and c1[i]+c3[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r13=1
        if(matchCount<=disCount):
                r13=-1
        rs13=max(matchCount,disCount)
        matchCount=0
        disCount=0
        for i in range(Gr.n):
                if c2[i]!=None and c3[i]!=1 and c2[i]==c3[i]:
                        matchCount=matchCount+1
                if c2[i]!=None and c3[i]!=1 and c2[i]+c3[i]==1:
                        disCount=disCount+1
        if(matchCount>disCount):
                r23=1
        if(matchCount<=disCount):
                r23=-1
        rs23=max(matchCount,disCount)
        if(r12*r13*r23==-1):
                if(rs12<rs13 and rs12<rs23):
                        r12=-r12
                else:
                        if(rs13<rs23):
                                r13=-r13
                        else:
                                r23=-r23
        output=[None]*Gr.n			
        for i in range(Gr.n):
                if(c1[i]!=None and c2[i]!=None and c3[i]!=None):
                        balance=c1[i]+r12*c2[i]+r13*c3[i]
                        if balance>0:
                                output[i]=1
                        else:
                                output[i]=0
        return output



#Uses classify2 to classify vertices from the graph stored in fileName.
#Then it refines the classification by switching vertices to the communities the majority of their neighbors are in.
#Finally, it checks how many vertices from each community it assigned to each community, assuming that the first 758 vertices are in one community and the rest are in the other.
def compAlgorithms(fileName):
        G=readGraph1(fileName)
        classes=[classify2(G)]
        for k in range(7):
                newClasses=[-1]*len(classes[0])
                for i in range(len(classes[0])):
                        diff=0
                        for j in G.edges[i]:
                                if classes[-1][j]==1:
                                        diff=diff+1
                                if classes[-1][j]==0:
                                        diff=diff-1
                        newClasses[i]=random.randint(0,1)
                        if classes[-1][i]==None:
                                newClasses[i]=None
                        if diff>0:
                                newClasses[i]=1
                        if diff<0:
                                newClasses[i]=0
                classes.append(newClasses)      
        count1=[0,0,0,0,0,0,0,0]
        count2=[0,0,0,0,0,0,0,0]
        for j in range(8):
                for i in range(758):
                        if classes[j][i]==0:
                                count1[j]=count1[j]+1
                        if classes[j][i]==1:
                                count2[j]=count2[j]+1   
                for i in range(758,G.n):
                        if classes[j][i]==1:
                                count1[j]=count1[j]+1
                        if classes[j][i]==0:
                                count2[j]=count2[j]+1   
        print ((count1,count2))
			
    
    
