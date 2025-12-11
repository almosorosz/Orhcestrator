#!/usr/bin/env python
# coding: utf-8

# In[1]:


def counter( folder, filename):
    import os 
    count = 0
    with os.scandir(folder) as entries:
        for entry in entries:           
            if filename in entry.name:
                count += 1
    return count


# In[ ]:




