//https://rust-unofficial.github.io/too-many-lists/second-generic.html

// making it generic
//

// use std::mem; // no more required

pub struct List<T> {
    head: Link<T>, // head can be either empty or a node
}
type Link<T> = Option<Box<Node<T>>>; // replacing enum which was basically an option to begin with

struct Node<T> {
    elem: T,
    next: Link<T>,
}

impl<T> List<T> {
    // list of functions that can be implemented on List

    // to create a new LL
    pub fn new() -> Self {
        List { head: None }
    }

    // to push value into the LL
    pub fn push(&mut self, element_to_be_pushed: T) {
        let new_node = Box::new(Node {
            elem: element_to_be_pushed,
            next: self.head.take(), // take is function form Option which is equivalen to mem::replace
        });
        self.head = Some(new_node);
    }

    // to pop
    pub fn pop(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;
            node.elem
        })
    }
}

// drop trait implementation
impl<T> Drop for List<T> {
    fn drop(&mut self) {
        let mut cur_link = self.head.take();
        while let Some(mut boxed_node) = cur_link {
            cur_link = boxed_node.next.take();
        }
    }
}

// testing
#[cfg(test)]
mod test {
    use super::List;

    #[test]
    fn basic() {
        // create a new list
        let mut list = List::new(); // List is in a different module
        assert_eq!(list.pop(), None); // to check the pop fucntion with empty list

        // populating the list
        list.push(10);
        list.push(1);
        list.push(0);

        // to check the pop fucntion with populated list
        assert_eq!(list.pop(), Some(0));
        assert_eq!(list.pop(), Some(1));
        assert_eq!(list.pop(), Some(10));
        assert_eq!(list.pop(), None);
    }
}
