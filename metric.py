def pascal_measure(gt, dt):
    # Ground Truth Bounding Box
    xmin1, ymin1, xmax1, ymax1 = gt[0], gt[1], gt[2], gt[3]
    a1 = (xmax1 - xmin1) * (ymax1 - ymin1)    

    # Detected Bounding Box
    xmin2, ymin2, xmax2, ymax2 = dt[0], dt[1], dt[2], dt[3]
    a2 = (xmax2 - xmin2) * (ymax2 - ymin2)   

    # Intersection of Bounding Boxes
    xmin3 = max(xmin1, xmin2), ymin3 = max(ymin1, ymin2), xmax3 = min(xmax1, xmax2), max3 = min(ymax1, ymax2)
    a3 = max(xmax3 - xmin3,0) * max(ymax3 - ymin3,0)
    
    # Pascal Measure
    r = a3 / (a1 + a2 - a3)
    return r

