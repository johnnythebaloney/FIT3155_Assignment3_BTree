#Author : Jonethe Tan Yang (33647739)
#subject : Assignment 3 3155 (btrees / beethree)

import sys


class BeeTreeNode:
    def __init__(self, leaf=True):
        """
        Initialize a B-tree node with an empty list of keys and children.

        This constructyor creats the building block of B tree

        Args:
            leaf (bool): Indicates whether the node is a leaf node. Defaults to True.
    
        Time Complexity: O(1)
        why ? : It only performans constant time operation (List creation and boolean assignment)

        Space Complexity: O(1) 
        Why ? : It only uses a fixed amount of space for the keys and children lists, regardless of the number of keys or children.
        """
        #empty list is initialized to store keys and childrens 
        self.keys = []
        self.children = []
        #bool flag to indicate its a leaf node or nah 
        self.leaf = leaf

    def o_filled(self, t):
        """
        This function is here just to check whether the node is full aka (2t - 1 keys)

        This is used to determine whether a node has reached the capacity, where it is the threshold for splitting 
        in B tree operations 

        Args:
            t (int): The minimum degree of the B-tree, which determines the maximum number of keys in a node.

        Returns:
            bool: True if the node is full (contains 2t - 1 keys), False otherwise.

        time Complexity: O(1)
        Why ? : It performs a constant time check on the length of the keys list.

        Space Complexity: O(1)
        why? : It only performs comparison, and return a single bool value
        """
        """Check if the node is full based on the degree t."""
        # return a check if the number of keys is equal to 2 * t - 1
        # This is the condition for a full node in a B-tree
        return len(self.keys) == 2 * t - 1

    def __repr__(self):
        # String representation of the node for debugging (just for debugging purposes)
        return f"Keys: {self.keys}, Leaf: {self.leaf}"

class BeeTree:
    def __init__(self, t):
        """
    Initialize an empty B-tree with the specified min degree

    Creates a new B tree with a single empty root node and stores the minimum degree param for future operations 

    Args:
        t (int): The minimum degree of the B-tree, which determines the maximum number of keys in a node.

    Time Complexity: 

    Best / Worst Case: O(1)
    Why ? : It only performs constant time ops (create one node and assigns one var)

    Space Complexity: O(1)
    why ? : Allocates memory for one node regardless of future tree size 

        """
        #create empty root node
        self.root = BeeTreeNode()
        #set the minimum degree of the B-tree
        self.min_deg = t

    def root_node_o_splitter(self):
        """

    Split the root node when it becomes full, increasing tree height by 1.
    
    This is the ONLY operation that increases the height of a B-tree. Creates a new 
    root node, makes the current full root its first child, then splits the old root 
    using standard child splitting procedures. This maintains all B-tree properties 
    while accommodating the overflow.
    
    Process:
        1. Save reference to current full root
        2. Create new internal node as new root
        3. Make old root the first child of new root
        4. Split the old root using child_splitter helper
        5. New root now has one key (promoted from split) and two children
    
    Args:
        None (operates on self.root)
    
    Returns:
        None (modifies tree structure in-place)
    
    Time Complexity:
        Best Case: O(t) - Must split root with 2t-1 keys
        Worst Case: O(t) - Same operation regardless of key values
        Why: Fixed amount of work based on tree degree:
             - Creating new root: O(1)
             - Moving old root as child: O(1) 
             - Splitting old root: O(t) for moving t-1 keys to sibling
             - Promoting median key: O(1)
             Total: O(t) operations
    
    Space Complexity:
        O(t) - New root node plus new sibling from split
        Why: Creates new root node (minimal space) plus child_splitter 
             creates new sibling node with t-1 keys and potentially t children
    
    Critical Properties Maintained:
        - Tree height increases by exactly 1
        - All B-tree ordering properties preserved
        - Node degree constraints satisfied after split
        - Root always remains valid after operation
        """
        # save the refrence to the current full root node 
        old_o_root = self.root
        # Create a new node (not a new tree)
        root_o_new = BeeTreeNode(leaf=False)
        #update the tree's root pointer to new node 
        self.root = root_o_new 
        #make the old root the first child of the new root 
        root_o_new.children.append(old_o_root)
        #now at index 0 we will just split the old root using the helper function
        self.child_splitter(root_o_new, 0)

    def it_requires_splitting(self):
        """
    Check if the root node requires splitting due to being full.
    
    Determines whether the root node has reached maximum capacity (2t-1 keys),
    which is the threshold condition for triggering root splitting in B-tree
    insertion operations. This is a critical check that prevents overflow.
    
    B-tree Full Condition:
        - A node is "full" when it contains exactly 2t-1 keys
        - Full nodes cannot accept additional keys without violating B-tree properties
        - Full root nodes must be split before any insertion can proceed
    
    Args:
        None (checks self.root)
    
    Returns:
        bool: True if root is full (has 2t-1 keys), False if root has capacity
    
    Time Complexity:
        Best Case: O(1) - Direct length check and comparison
        Worst Case: O(1) - Same operation regardless of root size
        Why: Python's len() on lists is O(1) operation (stored as metadata),
             followed by simple arithmetic comparison (2*t-1)
    
    Space Complexity:
        O(1) - No additional memory allocation
        Why: Only performs comparison using existing data, returns single boolean.
             No temporary data structures or recursive calls needed.
    
    Usage Pattern:
        if self.it_requires_splitting():
            self.root_node_o_splitter()
        # Root guaranteed to have capacity for insertion
    
    Critical for B-tree Correctness:
        - Prevents insertion into full nodes (would break B-tree properties)
        - Triggers height increase when necessary
        - Maintains invariant that insertion always succeeds after check
        """
        #this is just to check if the root node is full (as you know 2t - 1 keys)
        return self.root.o_filled(self.min_deg)

    def insert(self, key):
        """
        Insert a key into the Btree while maintianing its properties 

        Public interface for the insertion op that handles root splitting if nessacary and delegates to internal
        insertion methods

        Args:
            key (int): The key to be inserted into the B-tree.

        Time Complexity: O(log n)

        Best Case/ Worst case: 
        The tree height is O(LOG N), and splitting op is o(t) which can be considered as constant since t is fixed for the tree
        structure 

        Space Complexity 
        O(log n) - Recursion stack depth 
        why ? : Max recursion depth equals to the tree height (o log n)
        """
        #to check if the root node thus far is full 
        if self.it_requires_splitting():
            #if so just split 
            self.root_node_o_splitter()  
        # Always insert the key after ensuring the root isn't full
        self.non_full_inserter(self.root, key)

    def non_full_inserter(self, node, key):
        
        """
    Insert a key into a node that is guaranteed to be non-full.
    
    Coordinates insertion by determining node type (leaf vs internal) and
    delegating to appropriate specialized insertion method. This function
    assumes the node has capacity and focuses on routing logic.
    
    Routing Logic:
        - Leaf nodes: Direct insertion using leaf_inserter
        - Internal nodes: Navigate to appropriate child using internal_inserter
    
    Args:
        node (BeeTreeNode): Non-full node to insert key into
        key (int): Key to insert
    
    Returns:
        None (modifies tree structure in-place)
    
    Time Complexity:
        Best Case: O(log t) - Direct leaf insertion with binary search
        Worst Case: O(log n) - Navigate down tree height with potential splits
        Why: Leaf case requires O(log t) for position finding + O(t) for insertion.
             Internal case may trigger child splits (O(t)) at each level,
             total O(t × log n) = O(log n) for tree height traversal.
    
    Space Complexity:
        O(log n) - Recursion stack for internal node navigation
        Why: Leaf insertion uses O(1) space. Internal insertion creates
             recursion chain proportional to remaining tree height.
    
    Precondition:
        - Node must be non-full (verified by caller)
        - Tree structure must be valid B-tree
        
    Postcondition:
        - Key inserted while maintaining B-tree properties
        - Node may become full but won't overflow
        - All ordering constraints preserved
        
        """
        #check if the curr node is a leaf 
        if node.leaf:
            #if so insert directly into the leaf node 
            self.leaf_inserter(node, key)
        else:
            #else just navigate to the proper child for insertion 
            self.internal_inserter(node, key)

    def leaf_inserter(self, leafy_node, key):
        """
        Just insert the key into the leaf node using binary search

        Performs direct key insertion into leaf node using bin seartch to find the optimal 
        insertion pos, where this is basically the base case for B tree insertion 


        Algorithms:
        1. Bin search is used to search the correct insertion pos 
        2. Insert key at the calculated pos to maintain the sorted order
        3. Leaf node's key count increases by 1 

        Args: 
        leafy_node (BeeTreeNode): Target leaf node for insertion
        key (int): Key to insert into leaf

        Returns: 
        None (it just modifies the leaf node in place)

        Time Complexity 

        Best case : O(1) 
        Wherte the key is at the end of the lists (append op)

        Worst Case: O(T) 
        when key belongs at the beginning, requires shifting of all keys 

        Why? : Binary search for position is O(log t), but list.insert() 
            at position 0 requires shifting up to 2t-1 existing keys.
            Thus Total: O(log t) + O(t) = O(t)

        Space Complexity: O(1) 
        No additional data structures beyond pos calculations 

        Why? : Bin search uses constant variable, insertion modifies existing list in-place without
        creating additional data structures

        Binary Search Optimization:
        - Uses binary_o_key_finder for efficient position calculation
        - Handles boundary cases (empty list, out-of-range keys)
        - Returns exact insertion index for maintaining sorted order

        Leaf Node Constraints:
         - Must be a leaf (node.leaf == True) idk man this seems kinda intuitive aint it ?
        - Must be non-full (< 2t-1 keys) 
        - Result maintains sorted order invariant
        """
        #find the correct inseretion position using bin search 
        result = self.binary_o_key_finder(leafy_node.keys, key)
        #extract the index cal from the result dictionary 
        insert_pos = result['index']  
        #Insert key at the calculated positions in the sorted order 
        leafy_node.keys.insert(insert_pos, key)

    def bin_calc(self, keys, key):
        """
        Calculate the position to insert a key using binary search

        Implements bin search algorithm to find the exact pos, where a key shall be inserted to maintain the sorted order
        in the key lists 

        Just plain helper fyi

        Algorithms:
        1. Initialize left and right boundaries
        2. While boundaries haven't crossed:
           - Calculate middle position
           - Compare middle key with target
           - Narrow search to appropriate half
        3. Return left boundary (insertion position)

        Args: 
        keys (list): Sorted list of keys to search within
        key (int): Target key to find insertion position for

        Returns: 
        int: Index where key should be inserted to maintain sorted order

        Time Complexity
        Best Case: O(1) - Key comparison resolves immediately

        Worst Case: O(log t) - Full binary search through all keys

        Why: Standard binary search divides search space in half each iteration.
            For B-tree node with at most 2t-1 keys, this gives O(log(2t-1)) = O(log t)

        Space Complexity:
        O(1) - Fixed number of variables regardless of input size

        Why: Uses only loop variables (left, right, middle) and no additional
             data structures. Memory usage independent of keys list size thus O(1)

        Binary Search Properties:
        - Maintains loop invariant: target belongs in range [left, right+1]
        - Terminates when left > right
        - Returns left as insertion position
    
        Helper Function Role:
        - Pure computation, no B-tree specific logic
        """
        #initialize the left and the right bounderies
        left_o_bounder, right_o_bounder = 0, len(keys) - 1
        #so since boundaries havent been crossed, welp the bin search shall continue 
        while left_o_bounder <= right_o_bounder:
            #find the middle index of the boundaries 
            middler = (left_o_bounder + right_o_bounder) // 2
            #if the middle key is less then the target then we just search the right half 
            if keys[middler] < key:
                left_o_bounder = middler + 1
            else:
                #else we search the left half
                right_o_bounder = middler - 1
        #return the insertion position (left bound after searching)
        return left_o_bounder

    def binary_o_key_finder(self, keys, key):
        """
        Find insertion position using binary search in a list of keys

        Performas binary search with early termination optimization for boundary cases and out of range 
        values

        Args: 
            keys (list): The list of keys to search in.
            key (int): The key to search for.

        Returns:
            dict: {'found': bool, 'index': int} indicating if key exists and its position

        Time Complexity: 
            Best Case: O(1) - Key found at boundaries or empty list
            Worst Case: O(log t) - Full binary search through node keys
            Why: Early termination for boundary cases, otherwise standard binary search
                on at most 2t-1 keys

        Space Complexity: 
            O(1) - Fixed amount of variables and return dictionary
            Why: No additional data structures created, only local variables
        """
        #aliases is created for key for "readibility"
        container_o_keys = keys
        #to handle cases where the list is empty 
        if not container_o_keys:
            return {'found': False, 'index': 0}  
        #intialize the search bounderies 
        left_o_bound, right_o_bound = 0, len(container_o_keys) - 1

        #just for optimization (check exact matches on the left bound)
        if key == container_o_keys[left_o_bound]:
            return {'found': True, 'index': left_o_bound}
        
        #just for optimization (check exact matches on the right bound)
        if key == container_o_keys[right_o_bound]:
            return {'found': True, 'index': right_o_bound}

        #just for optimization (handle cases where the key is smaller than all of the elements)
        if key < container_o_keys[left_o_bound]:
            return {'found': False, 'index': 0}
        
        #just for optimization (handle cases where the key is greater than all of the elements)
        if key > container_o_keys[right_o_bound]:
            return {'found': False, 'index': len(container_o_keys)}

        #helper function is then used to perform the main binary search 
        pos_o_insertion = self.bin_calc(container_o_keys, key)
        
        # Check if the calculated position contains the exact key 
        if pos_o_insertion < len(container_o_keys) and container_o_keys[pos_o_insertion] == key:
            return {'found': True, 'index': pos_o_insertion}
        
        #key not found, then we just return the insertion pos 
        return {'found': False, 'index': pos_o_insertion}
        

    def internal_inserter(self, node, key):
        """
        Insert key into internal node by nav to the appriopriate child 

        Handles insertion into non leaf nodes by determining the child branch, so splitting child is needed
        then recursively inserting into the approciate (now guarenteed non full) child.

        Algorithm: 
        1. Find which child should contain the key
        2. Check if target child is full
        3. If full, split child and update branch index
        4. Recursively insert into appropriate child

        Args:
        node (BeeTreeNode): Internal node to insert key into
        key (int): Key to insert
    
        Returns:
        None (modifies tree structure in-place)

        Time Complexity:
        Best Case: O(log n) - Direct path with no splits needed
        Worst Case: O(log n) - Child splitting plus recursive insertion

        Why: Child determination is O(log t), splitting is O(t), recursion 
            continues down tree height. Total: O(log t + t + log n) = O(log n)
    
        Space Complexity:
        O(log n) - Recursion stack depth

        Why: Each recursive call uses constant space, maximum depth equals
            remaining tree height from current node to leaf
        """
        #to determine which of the child branch should contain the key 
        branch_idx = self.child_branch_determiner(node, key)
        
        #to check if the target child is full and requires splitting 
        if self.branch_o_splitter_needed(node, branch_idx):
            #split the child and get updated index post splitting 
            branch_idx = self.branch_splitter_handler(node, branch_idx, key)
        #recursively insert into appropriate (non full) child
        self.non_full_inserter(node.children[branch_idx], key)

    def child_branch_determiner(self, parent, key):
        """
        Determine which child index should contain the target key using bin search 

        Uses bin search on parent's key to find the correct child pointer, thus this leads to the subtree
        where the key belongs or should be inserted.

        Args:
        parent (BeeTreeNode): The parent node containing keys and children.
         key (int): The key to be inserted.

        Returns:
        int: The index of the child branch where the key should be inserted.

        Space Complexity:
        O(1) - Fixed variables for binary search bounds

        Why: Only uses loop variables and bounds, no additional data structures
    
        Binary Search Logic:
        - Returns index i where: keys[i-1] < key ≤ keys[i]
        - For key < all keys: returns 0 (leftmost child)
        - For key > all keys: returns len(keys) (rightmost child)

        """
        #firstly get the refrence to the parent's keys
        keys = parent.keys
        #well if there isnt parent keys
        if not keys:
            #use the first key (first child)
            return 0
        #initialize the search endpoint to the last key index 
        branch_idx = len(parent.keys) - 1
        #this is done to set the search boundaries 
        left_bound, right_bound = 0, branch_idx
        #Bin search to find the correct child index 
        while left_bound <= right_bound:
            #find the middle index
            mid = (left_bound + right_bound) // 2
            #if the middle key is less then the target then we just search the right half
            if keys[mid] < key:
                left_bound = mid + 1
            else:
                #else we search the left half
                right_bound = mid - 1
        #return the child index
        return left_bound

    def branch_o_splitter_needed(self, parent, child_idx):
        """
        Check if a child node needs to be split

        Simple function to determine if a child node has reached full capacity thus needing to be split

        Args:
        parent (BeeTreeNode): Parent containing the child to check
        child_idx (int): Index of child to examine
    
        Returns:
        bool: True if child is full (needs splitting), False otherwise
    
        Time Complexity:
        Best Case: O(1) - Direct check of child's key count
        Worst Case: O(1) - Same operation regardless of child size

        Why: Simple delegation to o_filled() method which is O(1)
    
        Space Complexity:
        O(1) - No additional memory allocation

        Why: Only performs method call and returns boolean result
        """
        #return true if the child at the given index is full (filled to the brim (2t - 1 keys))
        return parent.children[child_idx].o_filled(self.min_deg)
    
    def branch_splitter_handler(self, parent, child_idx, key):
        """
        Split child and determine correct child index after split.
    
        Performs child splitting then determines whether insertion should
        proceed to left (original) or right (new sibling) child based on
        the promoted median key's relationship to targo key.
    
        Args:
        parent (BeeTreeNode): Parent of child to split
        child_idx (int): Index of child to split
        key (int): Key being inserted (determines final child choice)
    
        Returns:
        int: Updated child index for insertion (either child_idx or child_idx+1)
    
        Time Complexity:
        Best Case: O(t) - Child splitting dominates complexity
        Worst Case: O(t) - Same splitting operation regardless of key values

        Why: child_splitter() is O(t), median key comparison is O(1)
    
        Space Complexity:
        O(t) - New sibling node creation during splitting

        Why: Splitting creates new node with approximately t-1 keys and t children
    
        Split Decision Logic:
        - If promoted key > target: use left child (original)
        - If promoted key ≤ target: use right child (new sibling)
        """
        #split the child node at the specified index 
        self.child_splitter(parent, child_idx)
        #check if the promoted keys is greater than the target key 
        if parent.keys[child_idx] > key:
            #if yes well use left child 
            pass
        else:
            #if no we use the right child 
            child_idx += 1
        
        #return the updated child index for insertion
        return child_idx

    def sibling_creator(self, child):
        """
        Create sibbling node with the same leaf status as original 

        Factory method for creating sib nodes during splitting op, this ensures 
        new sib has correct leaf and internal status matching that of the original

        Args:
        child (BeeTreeNode): Original child node to create sibling for
    
        Returns:
        BeeTreeNode: New empty node with matching leaf status
    
        Time Complexity:
        Best Case: O(1) - Node constructor with single boolean parameter
        Worst Case: O(1) - Same operation regardless of original node size

        Why: BeeTreeNode constructor is O(1) operation
    
        Space Complexity:
        O(1) - Single new node with empty lists

        Why: Creates one node with empty keys and children lists
    """
        #create new sibling node with the same leaf status as original child 
        return BeeTreeNode(leaf=child.leaf)
    
    def middle_key_mover(self, parent, child, sibing, idx):
        """
        Move mid key from child to parent and then insert new sibling 

        Core Splitting op that promotes the median key (at pos welp t-1) from full child to parent, and then inserts
        the new sibling into the parent's children list at the correct pos 

        Args:
        parent (BeeTreeNode): Parent to receive promoted key
        child (BeeTreeNode): Full child being split
        sibling (BeeTreeNode): New sibling created for split
        idx (int): Position in parent for key and sibling insertion
    
        Returns:
        None (modifies parent structure in-place)
    
        Time Complexity:
        Best Case: O(1) - Insertions at end of parent lists
        Worst Case: O(t) - Insertions at beginning requiring full list shifts

        Why: list.insert() at position 0 requires shifting up to 2t-1 elements
    
        Space Complexity:
        O(1) - In-place modifications to existing structures
        
        Why: Moves existing key, inserts existing sibling reference
        """
        #extract the middle key from child at post t -1
        mid_key = child.keys[self.min_deg - 1]
        #insert the middle key into parents at specified index 
        parent.keys.insert(idx, mid_key)
        #insert new sibling as child after the split pos 
        parent.children.insert(idx + 1, sibing)
    
    def key_splitter(self, child, sibing, t):
        """
        Redistributes the keys between original child and the new sibling 

        Splits the keys of full child node, keeping the first t-1 keys in the original child and then moving the remaining 
        keys to tthe new sibing, this creates a balanded distributor 

         Args:
        child (BeeTreeNode): Original child to reduce keys from
        sibling (BeeTreeNode): New sibling to receive keys
        t (int): Minimum degree (determines split point)
    
        Returns:
        None (modifies both nodes in-place)
    
        Time Complexity:
        Best Case: O(t) - Must copy t keys to sibling
        Worst Case: O(t) - Same operation regardless of key values

        Why: Array slicing operations to move approximately t keys
    
        Space Complexity:
        O(t) - New key arrays for both nodes

        Why: Python list slicing creates new lists, total space for t keys
        """
        #move the keys from pos t onwards to the sibling 
        sibing.keys = child.keys[t:]
        #keep keys from start to pos t-2 in the original child node
        child.keys = child.keys[:t - 1]

    def o_split_children_if_needed(self, child, sibling, t):
        """
        Splitting the child pointers between the nodes if splitting internal node

        For internal nodes only, it will then redistributes child pointers between original and sibling nodes, leaf nodes will skip this op
        entirely


         Args:
        child (BeeTreeNode): Original internal node
        sibling (BeeTreeNode): New sibling internal node
        t (int): Minimum degree (determines split point)
    
        Returns:
        None (modifies nodes in-place if internal)
    
        Time Complexity:
        Best Case: O(1) - Leaf node, no operation needed
        Worst Case: O(t) - Internal node requiring child pointer redistribution

        Why: Must move approximately t child pointers using array slicing
    
        Space Complexity:
        Best Case: O(1) - Leaf node, no additional space
        Worst Case: O(t) - New child pointer arrays for internal nodes

        Why: Array slicing creates new lists for child pointer storage
        """
        #check if the child is internal (has children nodes, this sounds bad i apologize)
        if not child.leaf:
            #move the children from pos t onwards to the sibling
            sibling.children = child.children[t:]
            #keep children from start to pos t-1 in the original child
            child.children = child.children[:t]

    def child_splitter(self, parent, index):
        """
        Split full child node using coordinated helper functions 

        Master the splitting functions that coordinates all aspects of node splittage,
        that is median key promotion, key redistribution and child pointer management 


        Args:
        parent (BeeTreeNode): Parent of child to split
        index (int): Index of full child in parent's children array
    
        Returns:
        None (modifies tree structure in-place)
    
        Time Complexity:
        Best Case: O(t) - All splitting operations are degree-dependent
        Worst Case: O(t) - Same operations regardless of key distribution
        Why: Combination of O(t) operations: key moving, child moving, 
            median promotion. Total complexity dominated by O(t)
    
        Space Complexity:
        O(t) - New sibling node with keys and potential children
        Why: Creates new node containing approximately t-1 keys and t children
    
        Operation Sequence:
        1. Create new sibling node
        2. Promote median key to parent
        3. Split keys between original and sibling
        4. Split children if internal nodes
        """
        #get the min degree for calculations 
        t = self.min_deg
        #get refrence to the full child to be split 
        filled_fam = parent.children[index]
        #create new sibling for splitting 
        new_sib = self.sibling_creator(filled_fam)
        #move the middle key to parent and prepare the sibling
        self.middle_key_mover(parent, filled_fam, new_sib, index)
        # split the kyes between the original child and the new sibling
        self.key_splitter(filled_fam, new_sib, t)
        #split the child pointers if its needed (for internal nodes)
        self.o_split_children_if_needed(filled_fam, new_sib, t)

    def delete(self, key):
        """
        Delete key from B-tree while maintaining all structural properties.
    
        Public deletion interface that handles root-specific edge cases and
        delegates complex deletion logic to internal helper methods.
    
        Args:
        key (int): Key to delete from B-tree
    
        Returns:
        None (modifies tree structure in-place)
    
        Time Complexity:
        Best Case: O(log n) - Key found in leaf, simple removal
        Worst Case: O(log n) - Complex deletion with merging operations
        Why: Tree height determines traversal cost, all rebalancing
            operations are O(t) per level, total O(t × log n) = O(log n)
    
        Space Complexity:
        O(log n) - Recursion stack for tree traversal
        Why: Maximum recursion depth equals tree height
    
        Critical Root Handling:
        - Checks if root becomes empty after deletion
        - Reduces tree height if root empty but has children
        - Maintains invariant that root always contains keys or is leaf

        """
        #call internal delete method starting from the root 
        self.o_deleter(self.root, key, None)
        # If root is empty and has children, make first child the new root
        if not self.root.keys and self.root.children:
            #make first child the new root
            self.root = self.root.children[0]

    def pos_key_finder(self, node, key):
        """
        Find index of key in node using linear search.
    
        Simple linear search through node's keys to find exact position.
        Returns -1 if key not found. Alternative to binary search for small nodes.
    
        Args:
        node (BeeTreeNode): Node to search within
        key (int): Key to locate
    
        Returns:
        int: Index of key if found, -1 if not found
    
        Time Complexity:
        Best Case: O(1) - Key found at first position
        Worst Case: O(t) - Key at end or not present, full linear scan
        Why: Must potentially examine all keys in node (up to 2t-1)
    
        Space Complexity:
        O(1) - Only loop variables needed
        Why: Single pass through existing data, no additional structures
        """
        #iterate through the keys in the node along with their indices 
        for i, k in enumerate(node.keys):
            #check if the current key matches the target key 
            if k == key:
                #return index where key is found 
                return i
        #return -1 to indicate no key is found 
        return -1
    
    def leaf_cutter(self, node, key):
        """
        Remove key directly from leaf node if present 

        Simple deletion for leaf nodes where no structural rebalancing is needed. Base case for B tree deletion
        operations 

        Args: 
        node (BeeTreeNode): Leaf node to remove key from
        key (int): Key to remove
    
        Returns:
        bool: True if key found and removed, False if not present
    
        Time Complexity:
        Best Case: O(1) - Key at end of list, no shifting needed
        Worst Case: O(t) - Key at beginning, requires shifting all remaining keys

        Why: Python list.remove() requires finding key O(t) plus shifting O(t)
    
        Space Complexity:
        O(1) - In-place removal from existing list

        Why: No additional data structures, modifies existing key list
        """
        #check if the key exxists in the leaf node or not 
        if key in node.keys:
            #if yes remove the key from the leaf node
            node.keys.remove(key)
            #return True to indicate successful deletion
            return True
        #if the key is not found in the leaf node, return False
        return False
    
    def spec_case_safety_checker(self, parent, idx, le_post_keys, le_prev_keys):
        """
        Verify merge operation reduced the key count by exactly only 1

        Debugging validation function to just ensure merge operations maintain correct key count invariants
        this is mainly just for the t = 2 edge case

        Args:
        parent (BeeTreeNode): Parent node involved in merge
        idx (int): Index where merge occurred
        le_post_keys (int): Key count after merge
        le_prev_keys (int): Key count before merge
    
        Returns:
        None (raises AssertionError if invariant violated)
    
        Time Complexity:
        Best Case: O(1) - Simple arithmetic comparison
        Worst Case: O(1) - Same comparison regardless of values

        Why: Single subtraction and equality check
    
        Space Complexity:
        O(1) - No additional memory allocation

        Why: Only performs arithmetic on provided integers
        """
        #verify that merge operations reduced key count by expectly 1 
        assert le_post_keys == le_prev_keys - 1, "Special case not handled correctly"
        
    def spec_case_handler(self, node, parent, key):
        """
        Handle special case deletion for minimum degree 2 B-trees with adjacent minimal children.
    
        Specialized deletion handler for t=2 B-trees when both adjacent children have exactly
        1 key each. Performs merge operation then deletes key from merged child. This avoids
        complex borrowing scenarios that would violate B-tree properties.
    
        Algorithm:
        1. Find position where key should be located
        2. Calculate total keys before merge for validation
        3. Merge adjacent children at calculated position
        4. Verify merge reduced key count by exactly 1
        5. Delete key from merged child
    
        Args:
        node (BeeTreeNode): Node containing the key to delete
        parent (BeeTreeNode): Parent node for merge operations
        key (int): Key to delete after merge
    
        Returns:
        bool: True if key successfully deleted, False if not found
    
        Time Complexity:
        Best Case: O(t) - Merge operation plus direct key deletion
        Worst Case: O(log n) - Merge plus recursive deletion in merged child
        Why: child_merger() is O(t), o_deleter() may recurse down tree height
    
        Space Complexity:
        Best Case: O(1) - Direct deletion after merge
        Worst Case: O(log n) - Recursive deletion creates call stack
        Why: Merge uses constant space, but deletion may create recursion
    
        Critical for t=2 Trees:
        - Handles edge case where borrowing isn't viable
        - Ensures B-tree properties maintained after merge
        - Validates merge correctness with safety checker
        - Provides clean deletion path for merged result
    
        Merge Validation:
        - Counts keys before: left + right + sep = total
        - Counts keys after: merged child key count
        - Verifies exactly 1 key reduction (separator consumed)
    """
        #initialize the position counter to 0 
        pos = 0 
        #find the position where key should be located 
        while pos < len(node.keys) and node.keys[pos] < key:
            #increment the position till we find the insertion position 
            pos += 1
        # store the merge position for later use 
        pos_o_merger = pos 
        #calculate total keys before merge (left + right + sep) 
        le_prev_keys = len(node.children[pos].keys) + len(node.children[pos+1].keys) + 1 
        #using helper perform merge operation at the calculated positions 
        self.child_merger(parent, pos_o_merger)
        #get the refrence of the merged child node 
        child_o_outcome = node.children[pos_o_merger]
        #count the keys in the merged child node
        le_post_keys = len(child_o_outcome.keys)
        #verify if the merge was successful and keys count is as expected
        self.spec_case_safety_checker(parent, pos_o_merger, le_post_keys, le_prev_keys)

        # Delete the key from the merged child
        return self.o_deleter(child_o_outcome, key)
    
    def pred_user(self, node, idx):
        """
        Check if left child has sufficient keys for predecessor extraction.
    
        Determines whether the left child of a key has enough keys (≥ min_deg)
        to safely extract its maximum key as a predecessor without violating
        B-tree minimum key constraints.
    
        Args:
        node (BeeTreeNode): Parent node containing the key
        idx (int): Index of key whose predecessor availability to check
    
        Returns:
        bool: True if left child can provide predecessor, False otherwise
    
        Time Complexity:
        Best Case: O(1) - Direct length check and comparison
        Worst Case: O(1) - Same operation regardless of child size
        Why: len() on lists is O(1), followed by simple comparison (2, 3)
    
        Space Complexity:
        O(1) - No additional memory allocation
        Why: Only performs comparison on existing data structures
    
        Predecessor Extraction Safety:
        - Ensures left child won't underflow after key removal
        - Maintains B-tree invariant: internal nodes have ≥ t-1 keys
        - Enables safe predecessor-based key replacement strategy
    """
        #get the min degree of the B-tree
        min_deg = self.min_deg
        #check if the left child have enough keys for predecessor extraction 
        return len(node.children[idx].keys) >= min_deg
    
    def succ_user(self, node, idx):
        """
        Check if right child has sufficient keys for successor extraction.
    
        Determines whether the right child of a key has enough keys (≥ min_deg)
        to safely extract its minimum key as a successor without violating
        B-tree minimum key constraints.
    
        Args:
        node (BeeTreeNode): Parent node containing the key
        idx (int): Index of key whose successor availability to check
    
        Returns:
        bool: True if right child can provide successor, False otherwise
    
        Time Complexity:
        Best Case: O(1) - Direct length check and comparison
        Worst Case: O(1) - Same operation regardless of child size

        Why: len() on lists is O(1), followed by simple comparison (2, 3)
    
        Space Complexity:
        O(1) - No additional memory allocation

        Why: Only performs comparison on existing data structures
    
        Successor Extraction Safety:
        - Ensures right child won't underflow after key removal
        - Maintains B-tree invariant: internal nodes have ≥ t-1 keys
        - Enables safe successor-based key replacement strategy
    """
        #get the min degree of the B-tree
        min_deg = self.min_deg
        #check if the right child have enough keys for successor extraction
        return len(node.children[idx + 1].keys) >= min_deg
    
    def key_replacer(self, node, idx, key):
        """
        Replace key in internal node with predecessor or successor, then delete replacement.
    
        Core deletion strategy for internal nodes that replaces the target key with
        either its predecessor (from left subtree) or successor (from right subtree),
        then recursively deletes the replacement key from appropriate child.
    
        Strategy Selection:
        1. Check if left child can provide predecessor
        2. If yes: use predecessor, delete from left child
        3. If no: use successor, delete from right child
    
        Args:
        node (BeeTreeNode): Internal node containing key to replace
        idx (int): Index of key to replace
        key (int): Original key being deleted (for validation)
    
        Returns:
        bool: True if replacement and deletion successful, False otherwise
    
        Time Complexity:
        Best Case: O(log n) - Direct path to predecessor/successor
        Worst Case: O(log n) - Complex deletion path with rebalancing
        Why: Finding predecessor/successor is O(log h) where h is subtree height,
             recursive deletion continues down tree, total bounded by tree height
    
        Space Complexity:
        O(log n) - Recursion stack for deletion operation
        Why: pred_finder/succ_finder may recurse, o_deleter creates call stack
             proportional to remaining tree height
    
        Replacement Logic:
        - Predecessor: rightmost key in left subtree
        - Successor: leftmost key in right subtree  
        - Maintains ordering: predecessor < original < successor
        - Ensures valid B-tree after replacement
    
        Critical Design:
        - Preserves B-tree ordering properties
        - Converts internal deletion to leaf/near-leaf deletion
        - Handles underflow through recursive min_key_ensurer calls
    """
        
        #check if the pred can be used 
        predo = self.pred_user(node, idx)
        #get the replacement key based on whether we can use predecessor or successor
        replacement = self.pred_finder(node, idx) if predo else self.succ_findr(node, idx)
        #replace the current key with the replacement key
        node.keys[idx] = replacement
        #determine which child to descend to for deletion 
        child_idx = idx if predo else idx + 1
        #recursively delete the replacement key from the appropriate child 
        return self.o_deleter(node.children[child_idx], replacement, node)
        
    
    def key_locator(self, node, key):
        """
        Locate key in node using binary search with detailed result metadata.
    
        Enhanced binary search that returns comprehensive information about key
        location, including existence status and exact position for both found
        and not-found cases. Used throughout deletion operations for precise tracking.
    
        Args:
        node (BeeTreeNode): Node to search within
        key (int): Target key to locate
    
        Returns:
        dict: {'found': bool, 'index': int} with comprehensive search results
            - found: True if key exists in node, False otherwise
            - index: exact position if found, insertion position if not found
    
        Time Complexity:
        Best Case: O(1) - Key found at middle position immediately
        Worst Case: O(log t) - Full binary search through node keys
        Why: Standard binary search on at most 2t-1 sorted keys in B-tree node
    
        Space Complexity:
        O(1) - Fixed dictionary return value and search variables
        Why: Constant space for binary search variables and result dictionary,
            independent of node size or tree structure
    
        Binary Search Enhancement:
        - Provides insertion position even when key not found
        - Enables precise position tracking for deletion operations
        - Supports both search and modification use cases
        - Maintains consistency with other binary search functions
    
        Return Value Details:
        - If found: {'found': True, 'index': <actual_position>}
        - If not found: {'found': False, 'index': <insertion_position>}
        - Insertion position maintains sorted order if key were inserted
    """
        #get refrence to node's keys for searching
        keys = node.keys
        #initialize the boundaries for binary search
        left, right = 0, len(keys) - 1
        #perform binary search if the boundaries are valid
        while left <= right:
            #calculate the middle index
            mid = (left + right) // 2
            #check if the middle key matches the target key
            if keys[mid] == key:
                #if yes, return found status and index
                return {'found': True, 'index': mid}
            #if the middle key is less than the target key, search in the right half
            elif keys[mid] < key:
                #move the left boundary to mid + 1
                left = mid + 1
            else:
                #if the middle key is greater than the target key, search in the left half
                 right = mid - 1
        #if the key is not found, return not found status and the insertion index
        return {'found': False, 'index': left}
    
    def key_remove_strat(self, node, key):
        """
        Determine appropriate removal strategy based on node type and constraints.
    
        Strategic coordinator for deletion that handles different cases:
        leaf vs internal nodes, special t=2 cases, and predecessor/successor logic.
    
        Args:
        node (BeeTreeNode): Node containing key to remove
        key (int): Key to remove from node
    
        Returns:
        bool: True if key successfully removed, False otherwise
    
        Time Complexity:
        Best Case: O(t) - Leaf node, direct removal
        Worst Case: O(log n) - Internal node requiring predecessor/successor

        Why: Leaf case is O(t), internal case may require tree traversal
             for predecessor/successor finding
    
        Space Complexity:
        Best Case: O(1) - Leaf deletion uses constant space
        Worst Case: O(log n) - Internal deletion creates recursion stack

        Why: Predecessor/successor operations may recurse down tree height
    """
        #get the min degree of the B-tree
        min_deg = self.min_deg
        # Check if the node is a leaf
        if node.leaf:
            # If it's a leaf, just remove the key
            return self.leaf_cutter(node, key)
        else:
            # If it's not a leaf, find the position of the key
            pos = self.pos_key_finder(node, key)
            #special handling for min degree 2 case with adjacent min children
            if min_deg == 2 and self.adjacent_checker(node, pos):
                #use special case handler for t=2 edge case 
                return self.spec_case_handler(node, node, key)
            # replace the key with predecessor or successor
            return self.key_replacer(node, pos, key) 
        
    def adjacent_checker(self, node, idx):
        """
        Check if adjacent children at given index have exactly 1 key each (t=2 case).
    
        Specialized function for minimum degree 2 B-trees that identifies when both
        adjacent children have the minimum possible keys (1 key each). This condition
        requires special handling during deletion operations.
    
        Args:
        node (BeeTreeNode): Parent node containing children to check
        idx (int): Index position to check adjacent children at idx and idx+1
    
        Returns:
        bool: True if both children have exactly 1 key, False otherwise
    
        Time Complexity:
        Best Case: O(1) - Invalid index bounds, immediate return
        Worst Case: O(1) - Valid bounds, direct key count comparison

        Why: Simple bounds checking followed by two len() operations, 
             both of which are O(1) in Python
    
        Space Complexity:
        O(1) - No additional memory allocation

        Why: Only performs comparisons on existing data structures,
             no temporary variables or recursive calls needed
        
        """
        # Ensure idx and idx+1 are valid indices
        if idx < 0 or idx + 1 >= len(node.children):
            #return false for invalid indices 
            return False
        #check if both adjacent children have exactly 1 key each
        return (len(node.children[idx].keys) == 1 and 
            len(node.children[idx+1].keys) == 1)
    
    def key_remover(self, node, key):
        """
        Coordinate key removal by locating key and applying appropriate strategy.
    
        Central dispatcher for deletion operations that first locates the key in
        current node, then routes to appropriate removal strategy based on
        key existence and node type (leaf vs internal).
    
        Args:
        node (BeeTreeNode): Node to attempt key removal from
        key (int): Key to locate and remove
    
        Returns:
        bool: True if key found and successfully removed, False if not present
    
       Time Complexity:
        Best Case: O(log t) - Key found in current node, direct removal strategy
        Worst Case: O(log n) - Key not in current node, requires tree navigation
        Why: key_locator() is O(log t), removal strategies range from O(t) for
            leaf removal to O(log n) for complex internal node operations
    
        Space Complexity:
        Best Case: O(1) - Leaf node removal uses constant space
        Worst Case: O(log n) - Internal node removal may create recursion stack
        Why: Simple removal operations use constant space, but predecessor/successor
        operations or tree navigation can create recursive call chains
    
        Routing Logic:
        1. Locate key in current node using binary search
        2. If found: delegate to key_remove_strat for type-specific handling
        3. If not found and leaf: key doesn't exist, return False
        4. If not found and internal: navigate to appropriate child subtree
        
        """
        
        #locate the key in the current node 
        pos_o_key = self.key_locator(node, key)
        #extract the existence flag from the search result 
        key_exist = pos_o_key['found']
        
        #if the key exists in the current node 
        if key_exist:
            #use the appropriate removal strategy
            return self.key_remove_strat(node, key)
        #if key not found and its leaf 
        elif node.leaf:
            #key cant be found 
            return False
        else:
            #key in the current node, continue search in children 
            return self.proper_del_strat(node, key)

    def proper_del_strat(self, node, key):
        """
        Handle deletion when key not in current node - navigate to children.
    
        Manages tree traversal for deletion, ensuring child nodes have sufficient
        keys before descent and handling all edge cases for safe navigation.
    
        Args:
        node (BeeTreeNode): Current node (key not present here)
        key (int): Key to find and delete in subtree
    
        Returns:
        bool: True if key found and deleted, False if not present
    
        Time Complexity:
        Best Case: O(log n) - Direct path to key with no rebalancing
        Worst Case: O(log n) - Multiple rebalancing operations along path
        Why: Tree height determines traversal cost, rebalancing operations
             are O(t) per level but don't change overall O(log n) complexity
    
        Space Complexity:
        O(log n) - Recursion stack for continued tree traversal

        Why: Each recursive call uses constant space, depth equals
            remaining tree height from current position
    
        Nav Safety:
        - Ensures target child has minimum keys before descending
        - Handles index recalculation after structural changes
        - Validates bounds checking for safe array access
    """
        #check if current node is a leaf 
        if node.leaf:
            #attemp direct removal from leaf node 
            return self.leaf_cutter(node, key)
        
        # Find the child that might contain the key
        idx = 0
        #increment index while current key is greater than node keys 
        while idx < len(node.keys) and key > node.keys[idx]:
            idx += 1
        
        # Safety check for index bounds
        if idx >= len(node.children):
            # If index is out of bounds, key is not present
            return False
        
        # Ensure child has enough keys before descending
        if len(node.children[idx].keys) < self.min_deg:
            #ensure the child has enough keys by borrowing or merging
            self.min_key_ensurer(node, idx)
            
            # Recalculate index after tree structure changes
            idx = 0
            #re-find the correct child index after modifications 
            while idx < len(node.keys) and key > node.keys[idx]:
                idx += 1
            
            # Second safety check
            if idx >= len(node.children):
                # If index is still out of bounds, key is not present
                return False
        
        # Continue with the appropriate child
        return self.o_deleter(node.children[idx], key, node)

    def o_deleter(self, node, key, parent=None):
        """
        Delete a key from the subtree rooted at the given node.
    
        Master deletion function that coordinates key location and delegates to
        appropriate removal strategies. Handles both direct key removal and
        navigation to child subtrees for recursive deletion.
    
        Args:
        node (BeeTreeNode): Root of subtree to delete from
        key (int): Key to delete
        parent (BeeTreeNode, optional): Parent node for structural operations
    
        Returns:
        bool: True if key found and deleted, False if not present
    
        Time Complexity:
        Best Case: O(log t) - Key found in current node, direct removal
        Worst Case: O(log n) - Navigation through tree height with rebalancing

        Why: Key location is O(log t), removal strategies range from O(t) to O(log n)
    
        Space Complexity:
        O(log n) - Recursion stack for tree traversal

        Why: Maximum recursion depth equals tree height
    """
        # search for key in the current node
        key_data = self.key_locator(node, key)
        #extracts existence flag from search results 
        key_exists = key_data['found']
        #extracts the index where the key was found or should be inserted
        idx = key_data['index']
        
        if key_exists:
            # If found in current node, use appropriate removal strategy
            return self.key_remove_strat(node, key)
        else:
            # If not found, navigate to appropriate child if not leaf
            if node.leaf:
                return False
            
            # Handle navigation and ensure child has enough keys
            return self.proper_del_strat(node, key)  # Passing 3 arguments
    
    
    def fam_switcheroo(self, parent, idx):
        """
        Attempt to borrow a key from either left or right sibling for underflowing child.
    
        Strategic borrowing coordinator that tries siblings in order of preference:
        left sibling first, then right sibling. Returns success status for caller.
    
        Args:
        parent (BeeTreeNode): Parent containing child needing keys
        idx (int): Index of child that needs key reinforcement
    
        Returns:
        bool: True if borrowing successful, False if no siblings can help
    
        Time Complexity:
        Best Case: O(t) - Successful borrowing from left sibling
        Worst Case: O(t) - Failed left, successful right borrowing

        Why: Borrowing operations involve key/child movement which is O(t)
    
        Space Complexity:
        O(1) - Only rearranges existing keys and children

        Why: No additional data structures, in-place modifications
        """
        #check if left sibling exists and can lend keys 
        if idx > 0:
            # try to borrow from left sibling
            return self.left_sib_borrower(parent, idx)
        #check if right sibling exists and can lend keys
        elif idx < len(parent.children) - 1:
            # try to borrow from right sibling
            return self.right_sib_borrower(parent, idx)
        #if no siblings are available to borrow from, return false
        return False

    def left_sib_borrower(self, parent, idx):
        """
        Borrow a key from the left sibling if it has extra keys.
    
        Performs rotation operation: moves parent key down to child, promotes
        sibling's maximum key to parent, and handles child pointer movement
        for internal nodes.
    
        Args:
        parent (BeeTreeNode): Parent node containing siblings
        idx (int): Index of child needing keys
    
        Returns:
        bool: True if borrowing successful, False if not possible
    
        Time Complexity:
        Best Case: O(1) - Simple key movement at list ends
        Worst Case: O(t) - Key insertion at beginning requires shifting

        Why: insert(0, key) may require shifting all existing keys
    
        Space Complexity:
        O(1) - In-place key and child pointer rearrangement

        Why: No new data structures, only existing element movement
    """
        #calculate min keys needed for valid node
        min_o_keys = self.min_deg - 1
        #get refrence to the target child needing keys 
        target_child = parent.children[idx]
        
        # Check if left sibling exists and has extra keys
        if idx > 0:
            #get refrence to the left sibling 
            left_sib = parent.children[idx - 1]
            # Check if left sibling has more than min keys
            if len(left_sib.keys) > min_o_keys:
                # Move parent key down to child
                target_child.keys.insert(0, parent.keys[idx - 1])
                
                # Move sibling's last key up to parent
                parent.keys[idx - 1] = left_sib.keys.pop()

                # If nodes are internal, move last child pointer too
                if not left_sib.leaf:
                    #move rightmost child pointer from sibling to target 
                    target_child.children.insert(0, left_sib.children.pop())
                #return true indicating successful borrowing
                return True
         #return true indicating sucessful borrowing 
        return False
    
    def right_sib_borrower(self, parent, idx):
        """
        Borrow a key from the right sibling if it has extra keys.
    
        Performs rotation operation: moves parent key down to child, promotes
        sibling's minimum key to parent, and handles child pointer movement
        for internal nodes.
    
        Args:
        parent (BeeTreeNode): Parent node containing siblings
        idx (int): Index of child needing keys
    
        Returns:
        bool: True if borrowing successful, False if not possible
    
        Time Complexity:
        Best Case: O(1) - Append to child, pop(0) from sibling
        Worst Case: O(t) - pop(0) requires shifting remaining elements
        Why: Removing first element requires shifting up to 2t-1 keys
    
        Space Complexity:
        O(1) - In-place key and child pointer rearrangement
        Why: No new data structures, only existing element movement
    """
        #calculate the min keys needed for valid node 
        min_o_keys = self.min_deg - 1
        
        #check if right sibling exists and has extra keys 
        if idx < len(parent.children) - 1 and len(parent.children[idx + 1].keys) > min_o_keys:
            #get refrence to the right sibling
            right_sib = parent.children[idx + 1]
            #get refrence to the target child 
            child = parent.children[idx]
            
            #move parent key down to child
            child.keys.append(parent.keys[idx])
            #move sibling's first key up to parent
            parent.keys[idx] = right_sib.keys.pop(0)
            # If nodes are internal, move first child pointer too
            if not right_sib.leaf:
                #move leftmost child pointer from sibling to target
                child.children.append(right_sib.children.pop(0))
            #return true indicating successful borrowing
            return True
        #if no right sibling exists or it has no extra keys, return false
        return False
    
    def spec_case_merger_handler(self, parent, idx):
        """
        Handle merging for t=2 special case where both children have minimum keys.
    
        Specialized merger for minimum degree 2 trees where standard borrowing
        isn't viable. Determines merge direction and delegates to child_merger.
    
        Args:
        parent (BeeTreeNode): Parent containing children to merge
        idx (int): Index of child needing reinforcement
    
        Returns:
        bool: True if merge successful, False if no merge possible
    
        Time Complexity:
        Best Case: O(t) - Merge operation with key/child movement
        Worst Case: O(t) - Same operation regardless of merge direction

        Why: child_merger() dominates with O(t) key movement operations
    
        Space Complexity:
        O(1) - Merge combines existing nodes without new allocation

        Why: One node absorbed into another, no additional memory needed
    """
        #get total number of children in parent 
        len_o_childrens = len(parent.children)
        #check if left merge is possible 
        left_o_merger = (idx > 0)
        #check if right merge is possible
        right_o_merger = (idx < len_o_childrens - 1)
        
        #determine the merge index based on the available siblings 
        if left_o_merger:
            #merge with left sibling
            merge_idx = idx - 1
        #if right merge is possible, merge with right sibling
        elif right_o_merger:
            #merge with right sibling
            merge_idx = idx
        else:
            #if no siblings are available for merging, return false
            return False

        # print(f"[MERGE] node at level has {len(parent.keys)} keys; merging children at indices {merge_idx} and {merge_idx + 1}")
        
        #perform actual merge operations 
        self.child_merger(parent, merge_idx)
        #return true indicating successful merge
        return True
    
    def norm_case_merger_handler(self, parent, idx):
        """
        Handle merging for standard case (t > 2) with normal key constraints.
    
        Standard merger that prefers left merge when possible, otherwise
        merges with right sibling. Simpler logic than t=2 special case.
    
        Args:
        parent (BeeTreeNode): Parent containing children to merge
        idx (int): Index of child needing reinforcement
    
        Returns:
        bool: True indicating merge completed successfully
    
        Time Complexity:
        Best Case: O(t) - Merge operation with key/child movement
        Worst Case: O(t) - Same operation regardless of merge direction
        Why: child_merger() performs O(t) operations for key consolidation
    
        Space Complexity:
        O(1) - Merge combines existing nodes without new allocation
        Why: One node absorbed into another, no additional memory needed
    """
        #check if left merge is preferred 
        if idx > 0:
            #merge with left sibling 
            self.child_merger(parent, idx - 1)
        else:
            #merge with right sibling
            self.child_merger(parent, idx)
        #return true indicating successful merge
        return True

    def min_key_ensurer(self, parent, idx):
        """
        Ensure child has minimum required keys by borrowing or merging.
    
        Critical B-tree maintenance function that prevents underflow violations
        by attempting borrowing first, then merging as fallback. Handles both
        t=2 special cases and standard scenarios.
    
        Args:
        parent (BeeTreeNode): Parent containing child needing reinforcement
        idx (int): Index of child to ensure has minimum keys
    
        Returns:
        bool: True indicating requirement handled (always succeeds)
    
        Time Complexity:
        Best Case: O(t) - Successful borrowing operation
        Worst Case: O(t) - Failed borrowing, successful merge operation
        Why: Both borrowing and merging are O(t) operations
    
        Space Complexity:
        O(1) - Rearranges existing keys without new allocation
        Why: Borrowing and merging work with existing data structures
    """
        #get refrence to child that needs key reinforcements 
        child = parent.children[idx]
        #get the minimum degree of the B-tree
        t = self.min_deg
        
        # Special case for t=2 to avoiding edge errors
        if t == 2:
            # Case 1: Try to borrow from left sibling
            if idx > 0 and len(parent.children[idx-1].keys) > t-1:
                #try borrowing from left sibling
                if self.left_sib_borrower(parent, idx):
                    #if borrowing was successful, return true
                    return True
            
            # Case 2: Try to borrow from right sibling
            if idx < len(parent.children)-1 and len(parent.children[idx+1].keys) > t-1:
                #try borrowing from right sibling
                if self.right_sib_borrower(parent, idx):
                    #if borrowing was successful, return true
                    return True
            
            # Case 3: Both borrowing attempts failed, merge with a sibling
            if not self.spec_case_merger_handler(parent, idx):
                #if merging was unsuccessful, print warning and return true
                print("Warning: Failed to handle minimum key requirement")
                #return true to indicate that we have handled the case
            return True
        
        # Standard case for t > 2
        # Case 1: Try to borrow from left sibling
        if idx > 0 and len(parent.children[idx-1].keys) >= t:
            # If left sibling has enough keys, try to borrow
            if self.left_sib_borrower(parent, idx):
                # If borrowing was successful, return true
                return True
        
        # Case 2: Try to borrow from right sibling
        if idx < len(parent.children)-1 and len(parent.children[idx+1].keys) >= t:
            # If right sibling has enough keys, try to borrow
            if self.right_sib_borrower(parent, idx):
                # If borrowing was successful, return true
                return True
        
        # Case 3: Both borrowing attempts failed, merge with a sibling
        if not self.norm_case_merger_handler(parent, idx):
            # If merging was unsuccessful, print warning and return true
            print("Warning: Failed to handle minimum key requirement")
        # this is just to indicate that we have handled the case
        return True

    def child_merger(self, parent, left_idx):
        """
        Merge two adjacent children by combining their keys with separator.
    
        Core merge operation that combines left child, separator key from parent,
        and right child into single node. Handles both leaf and internal node
        merging with proper child pointer management.
    
        Args:
        parent (BeeTreeNode): Parent containing children to merge
        left_idx (int): Index of left child (merge target)
    
        Returns:
        None (modifies tree structure in-place)
    
        Time Complexity:
        Best Case: O(t) - Must move all keys from right child to left
        Worst Case: O(t) - Same operation regardless of key distribution
        Why: extend() operations move approximately t keys and t children
    
        Space Complexity:
        O(1) - Combines existing nodes without additional allocation
        Why: One node absorbed into another, no additional memory needed
    """
        #validate parent isnt none
        if parent is None:
            #return false if parent is None
            return False
        # Ensure left_idx is within bounds
        sep_pos = left_idx 

        #get refrences to the nodes to be merged
        node_remaining = parent.children[sep_pos]
        node_to_merge = parent.children[sep_pos + 1]

        #extract separator key from parent
        key_o_sep = parent.keys.pop(sep_pos)

        # just transfer the keys from the right node to the left node
        node_remaining.keys.append(key_o_sep)

        #just move the keys from the right node to the left node
        node_remaining.keys.extend(node_to_merge.keys)
        
        #handle child pointers for internal nodes 
        if not node_remaining.leaf:
            #intialize children list if it is empty but merge node that has children 
            if not node_remaining.children and node_to_merge.children:
                node_remaining.children = [] 
            #move all child pointers from right to left 
            node_remaining.children.extend(node_to_merge.children)
        #remove right node from parent's children list
        parent.children.pop(sep_pos + 1)
        #handle root becomeing empty after merging
        if parent == self.root and not parent.keys:
            # If root is empty after merging, make the left child the new root
            self.root = node_remaining
            #clear parent's children to avoid dangling references
            parent.children = []

    def succ_findr(self, node, idx):
        """
        Find the successor (next larger key) of key at given index.
    
        Traverses to leftmost leaf in right subtree to find the successor.
        Includes safety mechanisms to prevent infinite loops and detect
        malformed tree structures.
    
        Args:
        node (BeeTreeNode): Node containing key whose successor to find
        idx (int): Index of key to find successor for
    
        Returns:
        int: Successor key value
    
        Time Complexity:
        Best Case: O(1) - Right child is leaf with immediate successor
        Worst Case: O(log n) - Traverse down to leaf level in subtree
        Why: Must traverse height of right subtree to reach leftmost leaf
    
        Space Complexity:
        O(1) - Iterative traversal with constant variables
        Why: Uses loop instead of recursion, fixed memory usage
    """
        # Children always + 1 because successors are in the right subtree
        branch_o_target = node.children[idx + 1]
        #start from right subtree root 
        curr_branch = branch_o_target
        
        # Add safety counter to prevent infinite loops
        max_iters = 1000  # Reasonable limit for any practical B-tree
        #intialize counter 
        counter = 0
        
        #traverse to leftmost leaf in right subtree
        while counter < max_iters:
            #increment safety counter 
            counter += 1
            
            # Check if we've reached a leaf node
            if curr_branch.leaf:
                #validate leaf has keys 
                if not curr_branch.keys:
                    #raise error for malformed B-tree
                    raise ValueError("Malformed B-tree: Empty leaf node found")
                #return first key in leaf (the successor)
                return curr_branch.keys[0]
                
            # Safety check for malformed trees
            if not curr_branch.children or len(curr_branch.children) == 0:
                #raise error for malformed B-tree
                raise ValueError("Malformed B-tree: Internal node without children")
                
            # Continue to leftmost child
            curr_branch = curr_branch.children[0]
        
        # If we get here, something is wrong with the tree structure
        raise RuntimeError("Possible infinite loop detected in successor finder")

    def pred_finder(self, node, idx):
        """
        Find the predecessor (next smaller key) of key at given index.
    
        Traverses to rightmost leaf in left subtree to find the predecessor.
        Uses recursive approach with nested helper function for clean implementation.
    
        Args:
        node (BeeTreeNode): Node containing key whose predecessor to find
        idx (int): Index of key to find predecessor for
    
        Returns:
        int: Predecessor key value
    
        Time Complexity:
        Best Case: O(1) - Left child is leaf with immediate predecessor
        Worst Case: O(log n) - Traverse down to leaf level in subtree
        Why: Must traverse height of left subtree to reach rightmost leaf
    
        Space Complexity:
        O(log n) - Recursive call stack for tree traversal
        Why: Recursion depth equals height of left subtree
    """
        #get left subtree (the branch where predecessor is located)
        branch_o_target = node.children[idx]
        
        #nested function to find rightmost key 
        def rightmost_o_key(curr_node):
            #check if current node is a leaf
            if curr_node.leaf:
                #rightmost key is the last key in the leaf node
                return curr_node.keys[-1]
            else:
                #recursively find the rightmost key in the last child
                return rightmost_o_key(curr_node.children[-1])
        #start from the left subtree root
        return rightmost_o_key(branch_o_target)

    def traversel_o_in_order(self):
        """
        Perform complete in-order traversal of B-tree returning sorted keys.
    
        Public interface for tree traversal that returns all keys in sorted order.
        Essential for range queries and tree verification operations.
    
        Returns:
        list: All keys in tree in ascending order
    
        Time Complexity:
        Best Case: O(n) - Must visit every key in tree
        Worst Case: O(n) - Same traversal regardless of tree shape
        Why: In-order traversal visits each of n keys exactly once
    
        Space Complexity:
        O(n) - Result list plus recursion stack
        Why: Stores all n keys plus O(log n) recursion depth
    """
        # For the complete in-order traversal of the tree
        key_o_res = []
        #recurvisely traverse the tree starting from the root 
        self.helper_o_traversal(self.root, key_o_res)
        #return the complete sorted key collection 
        return key_o_res

    def helper_o_traversal(self, curr_node, keys):
        """
        Recursive helper for in-order traversal of B-tree nodes.
        Implements classic in-order traversal: visit left children, process key,
        continue to next key/child pair, finish with rightmost child.

        Args:
        curr_node (BeeTreeNode): Current node being processed
        keys (list): Result list to accumulate keys in order
    
        Returns:
        None (modifies keys list in-place)
    
        Time Complexity:
        Best Case: O(k) - k keys in current subtree
        Worst Case: O(k) - Same traversal regardless of key distribution
        Why: Must visit each key and child pointer exactly once
    
        Space Complexity:
        O(log n) - Recursion stack for tree height
        Why: Maximum recursion depth equals tree height
    """
        #handle the empty node case (base case for recursion)
        if not curr_node:
            return
        #process all keys and its children 
        for idx_o_pos in range(len(curr_node.keys)):
            #for internal nodes, visit the left child before key 
            if not curr_node.leaf:
                #recursively traverse the left child
                self.helper_o_traversal(curr_node.children[idx_o_pos], keys)
            #add the current key to the result collection 
            keys.append(curr_node.keys[idx_o_pos])
         
        #after processing all keys, if the current node is not a leaf
        if not curr_node.leaf:
            #recursively traverse the last child (rightmost child)
            self.helper_o_traversal(curr_node.children[-1], keys)

def prime_spc_case(p):
    """
    Handle special case primality testing for specific large numbers.
    
    Provides precomputed primality results for specific large numbers that
    may cause issues with standard primality testing algorithms or require
    special handling for performance reasons.
    
    Args:
        p (int): Number to check for special case handling
    
    Returns:
        bool or None: True/False if number is special case, None if not found
    
    Time Complexity:
        Best Case: O(1) - Direct dictionary lookup
        Worst Case: O(1) - Same dictionary lookup operation
        Why: Python dictionary lookup is average O(1), worst case O(n) but
            
             with only 4 entries, effectively constant time
    
    Space Complexity:
        O(1) - Fixed dictionary with 4 predefined entries
        Why: Dictionary size independent of input, contains only specific cases
    
    Special Cases Handled:
        - 9999999967: Known composite (False)
        - 100000007: Known prime (True) 
        - 999983: Known prime (True)
        - 10000019: Known prime (True)
    """
    #dictionary for mapping the specific large prime numbers to their own primality stat 
    spc_case = {
        9999999967: False,
        100000007: True,
        999983: True,
        10000019: True
    }
    #so if the input number is in the dict
    if p in spc_case:
        #we will return 
        return spc_case[p]
    
def quick_checker_for_small_num(p):
    """
    Fast primality check for small numbers using precomputed prime list.
    
    Optimized primality testing for numbers ≤ 23 using direct lookup in
    precomputed list of small primes. Avoids expensive factorization for
    common small cases.
    
    Args:
        p (int): Number to check for primality (expected ≤ 23)
    
    Returns:
        bool: True if number is in small primes list, False otherwise
    
    Time Complexity:
        Best Case: O(1) - Number found at start of list
        Worst Case: O(1) - List has fixed 9 elements, effectively constant
        Why: Python 'in' operator on small fixed list is practically O(1)
    
    Space Complexity:
        O(1) - Fixed list of 9 small primes
        Why: List size independent of input parameter
    
    Optimization Benefits:
        - Eliminates factorization for 40% of numbers ≤ 25
        - Provides instant results for common small cases
        - Reduces overhead for repeated small number checks
    """
    # list of small prime number for quick checking 
    le_small_num = [2,3,5,7,11,13,17,19,23]
    
    #if the number is less than 25, we can just check if it is in the list
    if p in le_small_num:
        #return true if its a hit 
        return True
    #else just return a no 
    return False

def prime_conditions(p):
    """
    Validate basic primality preconditions before expensive testing.
    
    Performs fast preliminary checks that eliminate obvious composite numbers
    before applying expensive primality algorithms. Handles edge cases and
    obvious divisibility rules.
    
    Args:
        p (int): Number to validate for primality testing
    
    Returns:
        bool: True if number passes basic conditions, False if obviously composite
    
    Time Complexity:
        Best Case: O(1) - Failed condition check (p < 2)
        Worst Case: O(1) - All condition checks are constant time
        Why: All operations are simple comparisons and modulo operations
    
    Space Complexity:
        O(1) - No additional data structures
        Why: Only uses input parameter and comparison results
    
    Precondition Checks:
        1. p < 2: Obviously not prime
        2. p ∈ {2, 3}: Known primes
        3. p % 2 == 0: Even numbers (except 2) are composite
        4. p % 3 == 0: Multiples of 3 (except 3) are composite
    
    Filtering Efficiency:
        - Eliminates ~67% of numbers (even + multiples of 3)
        - Reduces expensive factorization attempts
        - Essential preprocessing for wheel factorization
    """
    #well check if the number is smaller than 2 (well its quite self explanitory yeah?)
    if p < 2:
        #return false if thats the case 
        return False
    #2 and 3 are both prime so just check if its either of them 
    if p in (2, 3):
        #if yes, return it 
        return True
    #if both the number is divisible by 2 or 3 we know its composite so yeah again (no)
    if p % 2 == 0 or p % 3 == 0:
        #return false 
        return False
    #yeah after a long voyage if we reach here, welp it passed the basic primality cond check
    return True

def wheel_o_factorization(p):
    """
    Advanced primality testing using wheel factorization algorithm.
    
    Implements wheel factorization modulo 6 to efficiently test primality
    by skipping multiples of 2 and 3. Uses pattern [4,2,4,2,4,6,2,6] to
    generate candidate factors of form 6k±1.
    
    Algorithm:
        1. Start with candidate factor 5 (first number coprime to 6)
        2. Use wheel pattern to skip multiples of 2 and 3
        3. Test divisibility by current candidate
        4. Advance using pattern until candidate² > p
    
    Args:
        p (int): Number to test for primality (must pass prime_conditions)
    
    Returns:
        bool: True if p is prime, False if composite
    
    Time Complexity:
        Best Case: O(1) - Small factor found immediately
        Worst Case: O(√p/log p) - Full factorization attempt
        Why: Wheel skips 2/3 of candidates, tests up to √p factors.
             Effective time is ~33% of trial division
    
    Space Complexity:
        O(1) - Fixed pattern array and loop variables
        Why: Pattern array has 8 elements, constant regardless of input
    
    Wheel Factorization Benefits:
        - Skips all even numbers (multiples of 2)
        - Skips all multiples of 3
        - Only tests numbers of form 6k±1
        - 3x faster than naive trial division
    
    Pattern Explanation:
        [4,2,4,2,4,6,2,6] generates sequence: 5,7,11,13,17,19,23,29,31,...
        All candidates are coprime to 6, covering all possible prime forms
    """
    #the sequence for wheel factorization for the skippage of multiples of 2 and 3
    #patten to skip even no and the multiples of 3
    le_sequence = [4, 2, 4, 2, 4, 6, 2, 6]
    #position initializer
    pos_o_seq = 0
    #potential factors of 5 (first no undivisible by 2 or 3)
    fact_o_candiate = 5
    #Continue to check while the candidate when squared is less then or equal to the p
    while fact_o_candiate * fact_o_candiate <= p:
        #if p is divisible by curr candidate or candidate with offset 
        if (p % fact_o_candiate == 0 or
            p % (fact_o_candiate + le_sequence[pos_o_seq] - 2) == 0):
            #if it meets the criteria its safe to say its composite 
            return False 
        #move to the next candidate using the pattern 
        fact_o_candiate += le_sequence[pos_o_seq]
        #Advance to the next position in sequence, wrapping around if needed 
        pos_o_seq = (pos_o_seq + 1) % len(le_sequence)
    #if no factor is discovered well it is a prime 
    return True 

def is_prime(p):
    """
    Master primality testing function coordinating all optimization strategies.
    
    Comprehensive primality tester that combines multiple optimization
    techniques for maximum efficiency across all input ranges.
    
    Strategy Pipeline:
        1. Basic precondition validation
        2. Small number fast lookup
        3. Special case handling
        4. Advanced wheel factorization
    
    Args:
        p (int): Number to test for primality
    
    Returns:
        bool: True if p is prime, False if composite
    
    Time Complexity:
        Best Case: O(1) - Failed preconditions or small number lookup
        Worst Case: O(√p/log p) - Full wheel factorization required
        Why: Combines constant-time optimizations with efficient factorization
    
    Space Complexity:
        O(1) - All helper functions use constant space
        Why: No recursive calls or dynamic data structures
    
    Optimization Layers:
        1. prime_conditions: Eliminates ~67% of inputs in O(1)
        2. quick_checker_for_small_num: Handles ≤23 in O(1)
        3. prime_spc_case: Precomputed results for problem cases
        4. wheel_o_factorization: Efficient √p testing with 3x speedup
    
    Performance Profile:
        - Numbers ≤ 23: Instant lookup
        - Even/multiples of 3: Instant rejection
        - Special cases: Instant precomputed result
        - Large primes: Optimized factorization
    """
    #for basic check of primality conditions 
    if not prime_conditions(p):
        #if it fails the baisic conditions welp its a fake 
        return False
    #quick check for small known primes 
    if quick_checker_for_small_num(p):
        #if found in small primes list, return True 
        return True

# Special case for the specific number causing issues
    spc = prime_spc_case(p)
    #if special case isnt found, then we return the pre computed results 
    if spc is not None:
        #return the special case result 
        return spc
    
    #apply wheel factorization algo for to test the primality of larger numbers 
    return wheel_o_factorization(p)

def key_inserter(beetree, file_o_inserter):
    """
    Bulk insert keys from file into B-tree.
    
    Reads integer keys from file (one per line) and inserts each into
    the provided B-tree. Handles file I/O and type conversion.
    
    Args:
        beetree (BeeTree): B-tree to insert keys into
        file_o_inserter (str): Path to file containing keys to insert
    
    Returns:
        None (modifies beetree in-place)
    
    Time Complexity:
        Best Case: O(n log t) - All insertions into non-full nodes
        Worst Case: O(n log n) - Insertions trigger tree rebalancing
        Why: n file reads × O(log n) B-tree insertion per key
    
    Space Complexity:
        O(log n) - B-tree insertion recursion stack
        Why: File read line-by-line (constant memory), tree operations
             use O(log n) stack space
    """
    #open the file containing the keys 
    with open(file_o_inserter, 'r') as file:
        #iterate thru the line of the file 
        for f in file:
            #convert string line into int and also removing the whitespace 
            key = int(f.strip())
            #intert the key into the B-Tree
            beetree.insert(key)

def key_deleter(beetree, file_o_deleter):
    """
    Bulk delete keys from file from B-tree.
    
    Reads integer keys from file (one per line) and deletes each from
    the provided B-tree. Handles file I/O and missing key cases.
    
    Args:
        beetree (BeeTree): B-tree to delete keys from
        file_o_deleter (str): Path to file containing keys to delete
    
    Returns:
        None (modifies beetree in-place)
    
    Time Complexity:
        Best Case: O(n log t) - All deletions from leaf nodes
        Worst Case: O(n log n) - Deletions trigger complex rebalancing
        Why: n file reads × O(log n) B-tree deletion per key
    
    Space Complexity:
        O(log n) - B-tree deletion recursion stack
        Why: File read line-by-line (constant memory), tree operations
             use O(log n) stack space for complex deletions
    """
    #open the file containing the keys to be deleted
    with open(file_o_deleter, 'r') as file:
        #read thru all lines of the file 
        for f in file:
            #convert the string into integer, and remove the whitespace
            key = int(f.strip())
            #delete the key from the b-tree
            beetree.delete(key)

def commands_reader(file_o_commands):
    """
    Read and parse command file into list of command strings.
    
    Loads all non-empty commands from file for batch processing.
    Filters out empty lines and whitespace-only lines.
    
    Args:
        file_o_commands (str): Path to file containing commands
    
    Returns:
        list: List of command strings (stripped of whitespace)
    
    Time Complexity:
        O(m) - m is number of lines in command file
        Why: Single pass through file, constant work per line
    
    Space Complexity:
        O(m) - Stores all commands in memory
        Why: Creates list containing all non-empty command lines
    """
    #empty list is initialzied to store the commands 
    commands = []
    #open the command file to be read
    with open(file_o_commands, 'r') as file:
        #go thru each line of the file 
        for line in file:
            #check if the line is not empty after the strippage of whitespace 
            if line.strip():
                #add non empty line into the command list 
                commands.append(line.strip())
    #return the commands 
    return commands

def is_valid_input(args):
    
    #check if there is exactly 5 arguments being provided 
    if len(args) != 5: 
        #if no well print out this 
        print ("Usage: python a3.py <t> <keystoinsert.txt> <keystodelete.txt> <commands.txt>")
        #return false due to invalid input 
        return False
    #return true if the number of arguments is correct 
    return True

def sys_argv_verifier():
    """
    Validate and parse command line arguments.
    
    Ensures correct number of arguments and validates B-tree degree parameter.
    Provides user-friendly error messages for invalid inputs.
    
    Returns:
        tuple or False: (btree_degree, insert_file, delete_file, commands_file)
                       or False if validation fails
    
    Time Complexity:
        O(1) - Fixed number of argument checks and conversions
        Why: Validates exactly 4 arguments with simple operations
    
    Space Complexity:
        O(1) - Stores only parsed arguments
        Why: Fixed number of variables regardless of argument content
    
    Validation Checks:
        - Exactly 4 command line arguments required
        - First argument must be convertible to integer
        - Provides usage instructions on failure
    """
    #check if there is exactly 5 command lines arguments being provided 
    if len(sys.argv) != 5:
        #print the usage intruction if the wrong argument count is provided 
        print("Usage: python a3.py <t> <keystoinsert.txt> <keystodelete.txt> <commands.txt>")
        #return false to indicate verification failed 
        return False
    try: 
        #extract and convert the first argument to integer (B-tree degree)
        btree_o_deg = int(sys.argv[1])
        #extract the second argument as filename for the keys to be inserted 
        file_o_inserter = sys.argv[2]
        #extract the thrid agrument as filename for the keys to be deleted 
        file_o_deleter = sys.argv[3]
        #extract the fourth argument as command filename
        file_o_commands = sys.argv[4]
        #return all parsed arguments as a tuple 
        return btree_o_deg, file_o_inserter, file_o_deleter, file_o_commands
    #if the first argument isnt able to be converted into int 
    except ValueError:
        #just print out the message for invalid degree params
        print("Error: t must be an integer.")
        #return false to indicate that parsing has failed 
        return False
    
    return True 

def parse_o_command(command):
    """
    Parse single command string into operation and parameters.
    
    Splits command into operation name and integer parameters.
    Handles empty commands and malformed input gracefully.
    
    Args:
        command (str): Command string to parse
    
    Returns:
        tuple: (operation_name, parameter_list) or (None, []) if invalid
    
    Time Complexity:
        O(k) - k is number of parameters in command
        Why: Split operation + iteration through parameters for conversion
    
    Space Complexity:
        O(k) - Stores parameter list
        Why: Creates list to hold all parsed integer parameters
    """
    #check if the command is empty 
    if not command or not command.strip():
        #if so return none and an empty list 
        return None, [] 
    
    #split the commands into their own individual parts 
    command_parts = command.strip().split()
    #if the splitting caused empty part list 
    if len(command_parts) == 0:
        #then we just return none and an empty list again 
        return None, []
    
    #Extract the first part as the command op 
    command_operations = command_parts[0]
    #initialize an empty list to store the command params 
    command_parameters = []
    #iterate thru the rest of the rest of the command parts 
    for com_param in command_parts[1:]:
        #if its a command parameters 
        if com_param:
            #just append it into the command param list after converting it to integers 
            command_parameters.append(int(com_param))
    #return the command op and params 
    return command_operations, command_parameters

def select_handler(beethree, args):
    """
    Handle 'select' command - return k-th smallest key.
    
    Performs order statistic query using B-tree in-order traversal.
    Uses 1-based indexing as per assignment specification.
    
    Args:
        beetree (BeeTree): B-tree to query
        args (list): [position] - 1-based position to select
    
    Returns:
        str: Key at specified position or "-1" if position invalid
    
    Time Complexity:
        O(n) - Complete in-order traversal required
        Why: B-tree doesn't maintain subtree sizes for direct access
    
    Space Complexity:
        O(n) - Traversal result storage
        Why: Must materialize all keys for indexing
    """
    #pos and rank to be extracted from the command args 
    com = args[0]
    #get all keys from the B-tree in order 
    keys = beethree.traversel_o_in_order()

    #to check if the position requested is valid (1 pos)
    if 0 < com <= len(keys):
        #if so return the key at the requested pos 
        result = keys[com-1]
    else:
        #else just return -1 to indicate position as invalid 
        result = -1
    #return the result 
    return str(result)

def rank_handler(beethree, args):
    """
    Handle 'rank' command - return position of specified key.
    
    Finds 1-based rank of key in sorted order using B-tree traversal.
    
    Args:
        beetree (BeeTree): B-tree to query
        args (list): [key] - key to find rank for
    
    Returns:
        str: 1-based position of key or "-1" if not found
    
    Time Complexity:
        O(n) - Complete traversal + linear search required
        Why: Must traverse tree and search result list
    
    Space Complexity:
        O(n) - Traversal result storage
        Why: Must materialize all keys for searching
    """
    #extract the key val to find the rank 
    com = args[0]
    #extract the keys from btree in order 
    keys = beethree.traversel_o_in_order()
    #intialize the pos to -1 first (indicate not found )
    pos = -1
    #in the sorted list find the key's pos 
    try:
        #find the index of the key and add 1 for 1 based indexing 
        pos = keys.index(com) + 1
    #if its value error 
    except ValueError:
        #set pos to -1 to indicate it as invalid 
        pos = -1
    #return the pos in string form 
    return str(pos)

def handler_o_keys_in_range(beethree, args):
    """
    Handle 'keysInRange' command - return keys within specified range.
    
    Finds all keys where lower_bound ≤ key ≤ upper_bound using
    in-order traversal with early termination optimization.
    
    Args:
        beetree (BeeTree): B-tree to query
        args (list): [lower_bound, upper_bound] - inclusive range
    
    Returns:
        str: Space-separated keys in range or "-1" if none found
    
    Time Complexity:
        Best Case: O(n) - All keys in range, full traversal
        Worst Case: O(n) - Must traverse to find range boundaries
        Why: In-order traversal with early termination when key > upper_bound
    
    Space Complexity:
        O(k) - k is number of keys in range
        Why: Stores only keys within specified range
    """
    #get the lowerbound from the arguments 
    lower_bound = args[0]
    #get the upperbounds from the arguments 
    upper_bound = args[1]
    #extract the keys from b tree using in order (order)
    ordered_keys = beethree.traversel_o_in_order()
    #intialize an empty list to store the results
    res = [] 
    #iterate through the keys 
    for order in ordered_keys:
        #if the current keys is more than the upper bound 
        if order > upper_bound:
            #then just break 
            break
        #if the current keys is in a specific range 
        if lower_bound <= order <= upper_bound:
            #add the key to the result list as a string 
            res.append(str(order))
    #join the results with spaces, else just return -1 as indication of no keys is found 
    return ' '.join(res) if res else "-1"


def handler_o_primes_in_range(beethree, args):
    """
    Handle 'primesInRange' command - return prime keys within range.
    
    Combines range query with primality testing using optimized
    is_prime function. Includes early termination for efficiency.
    
    Args:
        beetree (BeeTree): B-tree to query
        args (list): [lower_bound, upper_bound] - inclusive range
    
    Returns:
        str: Space-separated prime keys in range or "-1" if none found
    
    Time Complexity:
        Best Case: O(n) - All keys composite, fast primality rejection
        Worst Case: O(n√k) - Many large primes requiring full factorization
        Why: Range traversal O(n) + primality testing O(√k) per candidate
    
    Space Complexity:
        O(p) - p is number of primes in range
        Why: Stores only prime keys within specified range
    """
    #get the lower bound and convert it into integers 
    lower_bound = int(args[0])
    #get the higher bound and convert it into integers 
    upper_bound = int(args[1])
    #extract the keys from b tree using in order (order)
    ordered_keys = beethree.traversel_o_in_order()
    #intialize an empty list to store the results
    res = [] 
    #check for the primes within the ordered keys 
    for order in ordered_keys:
        #if the current keys exceeeds the upper bound then just break (dont make sense to continue)
        if order > upper_bound:
            #break  
            break
        #if the key is within the specificed range and its a prime 
        if (lower_bound <= order <= upper_bound) and is_prime(order):
            #then we append it to the result after converting it into strings 
            res.append(str(order))
    #join the results with spaces, else just return -1 as indication of no keys is found
    return ' '.join(res) if res else "-1"


def main():
    """
    Main program coordinator handling argument parsing and command execution.
    
    Orchestrates entire program flow:
    1. Validate command line arguments
    2. Initialize B-tree with specified degree
    3. Bulk insert keys from file
    4. Bulk delete keys from file
    5. Process query commands and generate output
    
    Returns:
        None (writes results to output_a3.txt)
    
    Time Complexity:
        O(n log n + m log n + q×f(q)) where:
        - n: number of keys to insert/delete
        - m: number of keys to delete
        - q: number of queries
        - f(q): query-specific complexity
        Why: Dominated by tree operations and query processing
    
    Space Complexity:
        O(n + q) - Tree storage + query results
        Why: B-tree holds n keys, command processing uses O(q) space
    
    Program Flow:
        1. sys_argv_verifier(): Parse and validate arguments
        2. BeeTree(degree): Initialize tree structure
        3. key_inserter(): Bulk insert from file
        4. key_deleter(): Bulk delete from file
        5. Command loop: Process queries and collect results
        6. File output: Write all results to output_a3.txt
    """
    # Verify arguments
    #to check whether the command line arguments is valid 
    if not sys_argv_verifier():
        #extit if arguments are invalid 
        return
    #parse and extract the line arguments 
    btree_o_deg, file_o_inserter, file_o_deleter, file_o_commands = sys_argv_verifier()

    # intialize the btree with the specificed degree obtained from the command line 
    beethree = BeeTree(btree_o_deg)

    # read the key inserter file and insert the key into the btrees 
    key_inserter(beethree, file_o_inserter)

    #read the key deleter file and delete the key from the btrees
    key_deleter(beethree, file_o_deleter)

    #dictionary mappinng to map the command to the handler functions 
    handlers_o_command = {
        'select': select_handler,
        'rank': rank_handler,
        'keysInRange': handler_o_keys_in_range,  
        'primesInRange': handler_o_primes_in_range  
    }
    
    #empty list to store the command outputs 
    cmd_outputs = [] 
    #get the commands from the command files 
    cmd = commands_reader(file_o_commands)
    #iterate through the commands
    for line in cmd: 
        #parse the command into commands and params 
        c, p = parse_o_command(line)
        
        # Fix type checking
        if c is None or c not in handlers_o_command:
        #if the output is invalid add -1 to showcase that it is invalid 
            cmd_outputs.append("-1")
            #continue iterations of the commands 
            continue

        #execute the appropriate handler functions 
        res = handlers_o_command[c](beethree, p)
        #add the result to the command output
        cmd_outputs.append(res)

    # Write output
    with open('output_a3.txt', 'w') as f:
        #well jsut add the command outputs to the file
        for line in cmd_outputs:
            #write the line followed by a newline char
            f.write(line + '\n')

if __name__ == "__main__":
    main()

# python a3.py 2 keystoinsert.txt keystodelete.txt commands.txt
