import torch

def _cov(x):
    assert x.ndim == 2
     
    mean = x.mean(dim=0)
    
    tmp = x - mean.unsqueeze(0)
    
    tmp = tmp.transpose(0, 1).mm(tmp)    
    
    return tmp / float(x.size(0)) # THIS is the BIASED version as in the paper (differs to numpy)


def CR_loss(X,Y):
    """
    DeCov loss from (Cogswell, ICLR '16)
    """
    
    cov_mat = _cov(X)    

    mask = torch.ones_like(cov_mat, dtype=torch.long)
    mask -= torch.eye(mask.size(0), device=cov_mat.device, dtype=torch.long)
    mask = mask.view(-1)
    i = mask.nonzero().squeeze()

    off_diagonal_part = cov_mat.view(-1).index_select(0, i)      
    off_diagonal_part = off_diagonal_part.pow(2)        

    return off_diagonal_part.sum()


def VR_loss(X, Y):
    """
    VR (variance-regularization) loss from (Choi, AAAI '19)
    """

    cov_mat = _cov(X)

    diagonal_part = torch.diag(cov_mat)        
    diagonal_part = diagonal_part.pow(2)        

    return diagonal_part.sum()


def cw_CR_loss(X, Y):
    """
    Classwise DeCov loss from (Choi, AAAI '19)
    """
    
    batch_labels = set(Y.cpu().tolist())
    
    loss = 0
    
    for y_i in batch_labels:
        
        X_y_i = X[y_i == Y]
        
        cov_mat = _cov(X_y_i)
                
        mask = torch.ones_like(cov_mat, dtype=torch.long)
        mask -= torch.eye(mask.size(0), device=cov_mat.device, dtype=torch.long)
        mask = mask.view(-1)
        i = mask.nonzero().squeeze()
        
        off_diagonal_part = cov_mat.view(-1).index_select(0, i)      
        off_diagonal_part = off_diagonal_part.pow(2)        
        
        loss += off_diagonal_part.sum()
        
    return loss


def cw_VR_loss(X, Y):
    """
    Classwise variance loss from (Choi, AAAI '19)
    """
    
    batch_labels = set(Y.cpu().tolist())
    
    loss = 0
    
    for y_i in batch_labels:
        
        X_y_i = X[y_i == Y]
        
        cov_mat = _cov(X_y_i)
        
        diagonal_part = torch.diag(cov_mat)        
        diagonal_part = diagonal_part.pow(2)        
        
        loss += diagonal_part.sum()
        
    return loss