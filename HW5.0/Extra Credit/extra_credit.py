import wikipedia

max_num_pages=10
topics=['joe biden', 'donald trump', 'andrew yang']

for topic in topics:
    titles=wikipedia.search(topic,results=max_num_pages)
    
    num_files=0
    sections=[]
    
    for title in titles:
    	try:
    		page = wikipedia.page(title, auto_suggest=False)
    		#print_info(page)
    
    		sections=sections+page.sections
    		num_files+=1
    	except:
    		print("SOMETHING WENT WRONG:", title);  
    
    #CONVERT TO ONE LONG STRING
    text=''
    for string in sections:
    	words=string.lower().split()
    	for word in words:	
    		if True:
    			text=text+word+'\n '
    
    text_file = open(topic+'.txt', "w")
    n = text_file.write(text)
    text_file.close()


