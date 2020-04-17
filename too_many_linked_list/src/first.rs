//https://rust-unofficial.github.io/too-many-lists/first.html

use std::mem;

pub struct List {
    head: Link, // head can be either empty or a node
}
enum Link {
    Empty,
    More(Box<Node>),
}

struct Node {
    elem: i32,
    next: Link,
}

impl List {
    // list of functions that can be implemented on List

    // to create a new LL
    pub fn new() -> Self {
        List { head: Link::Empty }
    }

    // to push value into the LL
    pub fn push(&mut self, element_to_be_pushed: i32) {
        let new_node = Box::new(Node {
            elem: element_to_be_pushed,
            next: mem::replace(&mut self.head, Link::Empty), // the old list gets pointed
        });
        self.head = Link::More(new_node);
    }

    // to pop
    pub fn pop(&mut self) -> Option<i32> {
        match mem::replace(&mut self.head, Link::Empty) {
            // if empty
            Link::Empty => None,
            Link::More(node) => {
                // it was supposed to be ref node, why???
                self.head = node.next;
                Some(node.elem)
            }
        }
    }
}

// drop trait implementation
impl Drop for List {
    fn drop(&mut self) {
        let mut cur_link = mem::replace(&mut self.head, Link::Empty);
        // `while let` == "do this thing until this pattern doesn't match"
        while let Link::More(mut boxed_node) = cur_link {
            cur_link = mem::replace(&mut boxed_node.next, Link::Empty);
            // boxed_node goes out of scope and gets dropped here;
            // but its Node's `next` field has been set to Link::Empty
            // so no unbounded recursion occurs.
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
