package example;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;

public class CoauthorPath {
    private Person path[];

    public CoauthorPath(Person p1, Person p2) {
        shortestPath(p1,p2);
    }

    public Person[] getPath() { return path; }

    private void tracing(int position) {
        Person pNow, pNext;
        int direction, i, label;

        label = path[position].getLabel();
        direction = Integer.signum(label);
        label -= direction;
        while (label != 0) {
            pNow = path[position];
            Person ca[] = pNow.getCoauthors();
            for (i=0; i<ca.length; i++) {
                pNext = ca[i];
                if (!pNext.hasLabel())
                    continue;
                if (pNext.getLabel() == label) {
                    position -= direction;
                    label -= direction;
                    path[position] = pNext;
                    break;
                }
            }
        }
    }

    private void shortestPath(Person p1, Person p2) {
        Collection<Person>  h,
                now1 = new HashSet<Person>(),
                now2 = new HashSet<Person>(),
                next = new HashSet<Person>();
        int direction, label, n;

        Person.resetAllLabels();
        if (p1 == null || p2 == null)
            return;
        if (p1 == p2) {
            p1.setLabel(1);
            path = new Person[1];
            path[0] = p1;
            return;
        }

        p1.setLabel( 1); now1.add(p1);
        p2.setLabel(-1); now2.add(p2);

        while (true) {
            if (now1.isEmpty() || now2.isEmpty())
                return;

            if (now2.size() < now1.size()) {
                h = now1; now1 = now2; now2 = h;
            }

            Iterator<Person> nowI = now1.iterator();
            while (nowI.hasNext()) {
                Person pnow = nowI.next();
                label = pnow.getLabel();
                direction = Integer.signum(label);
                Person neighbours[] = pnow.getCoauthors();
                int i;
                for (i=0; i<neighbours.length; i++) {
                    Person px = neighbours[i];
                    if (px.hasLabel()) {
                        if (Integer.signum(px.getLabel())
                                ==direction)
                            continue;
                        if (direction < 0) {
                            Person ph;
                            ph = px; px = pnow; pnow = ph;
                        }
                        // pnow has a positive label,
                        // px a negative
                        n = pnow.getLabel() - px.getLabel();
                        path = new Person[n];
                        path[pnow.getLabel()-1] = pnow;
                        path[n+px.getLabel()] = px;
                        tracing(pnow.getLabel()-1);
                        tracing(n+px.getLabel());
                        return;
                    }
                    px.setLabel(label+direction);
                    next.add(px);
                }
            }
            now1.clear(); h = now1; now1 = next; next = h;
        }
    }
}
