class EELS_edge:
    def __init__( self, label="", e_bsub=None, e_int=None ):
        self.label  = label
        self.e_bsub = e_bsub
        self.e_int  = e_int

    def from_KEM( self, edge_KEM ):
        # construct edge from KEM convention
        # edge_KEM=['label',bsub_start,bsub_end,int_start,int_end]
        self.label = edge_KEM[0]
        self.e_bsub = (edge_KEM[1], edge_KEM[2])
        self.e_int = (edge_KEM[3], edge_KEM[4])

    def __str__( self ):
        s = "{}:".format(self.label)
        if self.e_bsub is not None:
            s+= ", e_bsub ({},{})".format(*self.e_bsub)
        if self.e_int is not None:
            s+= ", e_int ({},{})".format(*self.e_int)
        return s

    def __repr__( self ):
        return self.__str__()
    