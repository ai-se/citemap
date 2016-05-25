package example;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.helpers.DefaultHandler;


public class Person {
    private static Map<String, Person> personMap = new HashMap<String, Person>();

    private String name;
    private String urlpt;

    /*
     * Create a new Person object.
     */
    private Person(String name, String urlpt) {
        this.name = name;
        this.urlpt = urlpt;
        personMap.put(name, this);
        coauthorsLoaded = false;
        labelvalid = false;
    }

    /*
     * Create a new Person object if necessary.
     */
    static public Person create(String name, String urlpt) {
        Person p;
        p = searchPerson(name);
        if (p == null)
            p = new Person(name, urlpt);
        return p;
    }

    /*
     * Returns the number of Person objects which
     * have been created until now.
     */
    static public int numberOfPersons() {
        return personMap.size();
    }

    /*
     * Check if a Person object already has been created.
     */
    static public Person searchPerson(String name) {
        return personMap.get(name);
    }

    /*
     * Coauthor information is loaded on demand only
     */

    private boolean coauthorsLoaded;
    private Person coauthors[];

    static private SAXParser coauthorParser;
    static private CAConfigHandler coauthorHandler;
    static private List<Person> plist = new ArrayList<Person>();

    static private class CAConfigHandler extends DefaultHandler {

        private String Value, urlpt;
        private boolean insideAuthor;

        public void startElement(String namespaceURI, String localName,
                                 String rawName, Attributes atts) throws SAXException {
            if (insideAuthor = rawName.equals("author")) {
                Value = "";
                urlpt = atts.getValue("urlpt");
            }
        }

        public void endElement(String namespaceURI, String localName,
                               String rawName) throws SAXException {
            if (rawName.equals("author") && Value.length() > 0) {
                plist.add(create(Value,urlpt));
                /* System.out.println(p + "   " + urlpt + "   " + plist.size()); */
            }
        }

        public void characters(char[] ch, int start, int length)
                throws SAXException {

            if (insideAuthor)
                Value += new String(ch, start, length);
        }

        private void Message(String mode, SAXParseException exception) {
            System.out.println(mode + " Line: " + exception.getLineNumber()
                    + " URI: " + exception.getSystemId() + "\n" + " Message: "
                    + exception.getMessage());
        }

        public void warning(SAXParseException exception) throws SAXException {

            Message("**Parsing Warning**\n", exception);
            throw new SAXException("Warning encountered");
        }

        public void error(SAXParseException exception) throws SAXException {

            Message("**Parsing Error**\n", exception);
            throw new SAXException("Error encountered");
        }

        public void fatalError(SAXParseException exception) throws SAXException {

            Message("**Parsing Fatal Error**\n", exception);
            throw new SAXException("Fatal Error encountered");
        }
    }

    static {
        try {
            coauthorParser = SAXParserFactory.newInstance().newSAXParser();

            coauthorHandler = new CAConfigHandler();
            coauthorParser.getXMLReader().setFeature(
                    "http://xml.org/sax/features/validation", false);

        } catch (ParserConfigurationException e) {
            System.out.println("Error in XML parser configuration: "
                    + e.getMessage());
            System.exit(1);
        } catch (SAXException e) {
            System.out.println("Error in parsing: " + e.getMessage());
            System.exit(2);
        }
    }

    private void loadCoauthors() {
        if (coauthorsLoaded)
            return;
        plist.clear();
        try {
            URL u = new URL("http://dblp.uni-trier.de/rec/pers/" + urlpt
                    + "/xc");
            coauthorParser.parse(u.openStream(), coauthorHandler);
        } catch (IOException e) {
            System.out.println("Error reading URI: " + e.getMessage());
            coauthors = new Person[0];
            return;
        } catch (SAXException e) {
            System.out.println("Error in parsing: " + name + " "+ e.getMessage());
            coauthors = new Person[0];
            return;
        }
        coauthors = new Person[plist.size()];
        coauthors = plist.toArray(coauthors);
        coauthorsLoaded = true;
    }

    public Person[] getCoauthors() {
        if (!coauthorsLoaded) {
            loadCoauthors();
        }
        return coauthors;
    }

    private int label;
    private boolean labelvalid;

    public int getLabel() {
        if (!labelvalid)
            return 0;
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
        labelvalid = true;
    }

    public void resetLabel() {
        labelvalid = false;
    }

    public boolean hasLabel() {
        return labelvalid;
    }

    static public void resetAllLabels() {
        Iterator<Person> i = personMap.values().iterator();
        while (i.hasNext()) {
            Person p = i.next();
            p.labelvalid = false;
            p.label = 0;
        }
    }

    public String toString() {
        return name;
    }

    /****
     private static int path1FirstStep(Person p1, Person p2) {
     Collection<Person> now = new HashSet<Person>();
     Collection<Person> next = new HashSet<Person>();

     int n = 1;

     p1.setLabel(n);
     now.add(p1);

     while (!now.isEmpty()) {
     n++;
     Iterator<Person> nowI = now.iterator();
     while (nowI.hasNext()) {
     Person pnow = nowI.next();
     Person neighbours[] = pnow.getCoauthors();
     int i;
     for (i=0; i<neighbours.length; i++) {
     Person px = neighbours[i];
     if (px.hasLabel())
     continue;
     px.setLabel(n);
     next.add(px);
     if (px == p2) {
     return n;
     }
     }
     }
     now.clear();
     Collection<Person> h = now;
     now = next;
     next = h;
     }
     return 0;
     }
     /****
     public static Person[] shortestPath1(Person p1, Person p2) {
     Person path[];
     int n, i;
     Person pNow, pNext;

     resetAllLabels();

     if (p1 == null || p2 == null)
     return null;
     if (p1 == p2) {
     n = 1;
     p1.setLabel(1);
     } else {
     n = path1FirstStep(p1, p2);
     }
     if (n < 1)
     return null;

     path = new Person[n];
     n--;
     path[n] = p2;
     while (n > 0) {
     pNow = path[n];

     Person ca[] = pNow.getCoauthors();
     for (i=0; i<ca.length; i++) {
     pNext = ca[i];
     if (!pNext.hasLabel())
     continue;
     if (pNext.getLabel()==n) {
     n--;
     path[n] = pNext;
     break;
     }
     }

     }
     path[0] = p1;
     return path;
     }

     private static void tracing(Person path[], int position) {
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


     private static Person middleP1, middleP2;

     private static boolean path2FirstStep(Person p1, Person p2) {
     Collection<Person> now1 = new HashSet<Person>();
     Collection<Person> now2 = new HashSet<Person>();
     Collection<Person> next1 = new HashSet<Person>();
     Collection<Person> next2 = new HashSet<Person>();

     int n = 1;

     p1.setLabel(n);
     now1.add(p1);
     p2.setLabel(-n);
     now2.add(p2);

     while (true) {
     if (now1.isEmpty() || now2.isEmpty())
     return false;
     System.out.println(n + ":");
     System.out.println("|now1| = " + now1.size());
     System.out.println("|now2| = " + now2.size());
     n++;
     Iterator<Person> nowI = now1.iterator();
     while (nowI.hasNext()) {
     Person pnow = nowI.next();
     Person neighbours[] = pnow.getCoauthors();
     int i;
     for (i=0; i<neighbours.length; i++) {
     Person px = neighbours[i];
     if (px.hasLabel()) {
     if (px.getLabel() > 0)
     continue;
     middleP1 = pnow;
     middleP2 = px;
     // System.out.println("<" + pnow + "," + px + ">");
     return true;
     }
     px.setLabel(n);
     next1.add(px);
     }
     }
     nowI = now2.iterator();
     while (nowI.hasNext()) {
     Person pnow = nowI.next();
     Person neighbours[] = pnow.getCoauthors();
     int i;
     for (i=0; i<neighbours.length; i++) {
     Person px = neighbours[i];
     if (px.hasLabel()) {
     if (px.getLabel() < 0)
     continue;
     middleP1 = px;
     middleP2 = pnow;
     // System.out.println("<" + pnow + "," + px + ">");
     return true;
     }
     px.setLabel(-n);
     next2.add(px);
     }
     }
     Collection<Person> h;
     now1.clear(); h = now1; now1 = next1; next1 = h;
     now2.clear(); h = now2; now2 = next2; next2 = h;
     }
     }

     public static Person[] shortestPath2(Person p1, Person p2) {
     Person path[];
     int n;
     resetAllLabels();

     if (p1 == null || p2 == null)
     return null;
     if (p1 == p2) {
     n = 1;
     p1.setLabel(1);
     path = new Person[1];
     path[0] = p1;
     return path;
     }
     if (!path2FirstStep(p1, p2))
     return null;
     n = middleP1.getLabel() - middleP2.getLabel();

     path = new Person[n];
     path[middleP1.getLabel()-1] = middleP1;
     path[n+middleP2.getLabel()] = middleP2;
     tracing(path,middleP1.getLabel()-1);
     tracing(path,n+middleP2.getLabel());
     return path;
     }


     public static Person[] shortestPath3(Person p1, Person p2) {
     Collection<Person> now1 = new HashSet<Person>();
     Collection<Person> now2 = new HashSet<Person>();
     Collection<Person> next = new HashSet<Person>();
     Collection<Person> h;
     Person path[];
     int direction, label, n;

     resetAllLabels();
     if (p1 == null || p2 == null)
     return null;
     if (p1 == p2) {
     n = 1;
     p1.setLabel(1);
     path = new Person[1];
     path[0] = p1;
     return path;
     }

     p1.setLabel( 1); now1.add(p1);
     p2.setLabel(-1); now2.add(p2);

     while (true) {
     if (now1.isEmpty() || now2.isEmpty())
     return null;

     if (now2.size() < now1.size()) {
     h = now1; now1 = now2; now2 = h;
     }
     // System.out.println("|now1| = " + now1.size());
     // System.out.println("|now2| = " + now2.size());

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
     if (Integer.signum(px.getLabel())==direction)
     continue;
     if (direction < 0) {
     Person ph;
     ph = px; px = pnow; pnow = ph;
     }
     // pnow has a positive label, px a negative
     // System.out.println("<" + pnow + "," + px + ">");
     n = pnow.getLabel() - px.getLabel();
     path = new Person[n];
     path[pnow.getLabel()-1] = pnow;
     path[n+px.getLabel()] = px;
     tracing(path,pnow.getLabel()-1);
     tracing(path,n+px.getLabel());
     return path;
     }
     px.setLabel(label+direction);
     next.add(px);
     }
     }
     now1.clear(); h = now1; now1 = next; next = h;
     }
     }


     *************/

    /*
     * publication information is loaded on demand only
     */

    private boolean publLoaded;
    private Publication publications[];
    private Publication personRecord;

    static private SAXParser publParser;
    static private PublConfigHandler publHandler;
    static private List<Publication> publlist = new ArrayList<Publication>();
    static private Publication hp;

    static private class PublConfigHandler extends DefaultHandler {

        private String Value;
        private boolean insideKey, insideHp;

        public void startElement(String namespaceURI, String localName,
                                 String rawName, Attributes atts) throws SAXException {
            if (insideKey = rawName.equals("dblpkey")) {
                Value = "";
                insideHp = (atts.getValue("type") != null);
            }
        }

        public void endElement(String namespaceURI, String localName,
                               String rawName) throws SAXException {
            if (rawName.equals("dblpkey") && Value.length() > 0) {
                Publication p = Publication.create(Value);
                if (insideHp)
                    hp = p;
                else
                    publlist.add(p);
            }
        }

        public void characters(char[] ch, int start, int length)
                throws SAXException {

            if (insideKey)
                Value += new String(ch, start, length);
        }

        private void Message(String mode, SAXParseException exception) {
            System.out.println(mode + " Line: " + exception.getLineNumber()
                    + " URI: " + exception.getSystemId() + "\n" + " Message: "
                    + exception.getMessage());
        }

        public void warning(SAXParseException exception) throws SAXException {

            Message("**Parsing Warning**\n", exception);
            throw new SAXException("Warning encountered");
        }

        public void error(SAXParseException exception) throws SAXException {

            Message("**Parsing Error**\n", exception);
            throw new SAXException("Error encountered");
        }

        public void fatalError(SAXParseException exception) throws SAXException {

            Message("**Parsing Fatal Error**\n", exception);
            throw new SAXException("Fatal Error encountered");
        }
    }

    static {
        try {
            publParser = SAXParserFactory.newInstance().newSAXParser();

            publHandler = new PublConfigHandler();
            publParser.getXMLReader().setFeature(
                    "http://xml.org/sax/features/validation", false);

        } catch (ParserConfigurationException e) {
            System.out.println("Error in XML parser configuration: "
                    + e.getMessage());
            System.exit(1);
        } catch (SAXException e) {
            System.out.println("Error in parsing: " + e.getMessage());
            System.exit(2);
        }
    }

    private void loadPublications() {
        if (publLoaded)
            return;
        publlist.clear();
        hp = null;
        try {
            URL u = new URL("http://dblp.uni-trier.de/rec/pers/" + urlpt
                    + "/xk");
            publParser.parse(u.openStream(), publHandler);
        } catch (IOException e) {
            System.out.println("Error reading URI: " + e.getMessage());
            coauthors = new Person[0];
            return;
        } catch (SAXException e) {
            System.out.println("Error in parsing: " + name + " "+ e.getMessage());
            coauthors = new Person[0];
            return;
        }
        publications = new Publication[publlist.size()];
        publications = publlist.toArray(publications);
        personRecord = hp;
        publLoaded = true;
    }

    public int getNumberOfPublications() {
        if (!publLoaded) {
            loadPublications();
        }
        return publications.length;
    }

    public Publication[] getPublications() {
        if (!publLoaded) {
            loadPublications();
        }
        return publications;
    }

    public Publication getPersonRecord() {
        if (!publLoaded) {
            loadPublications();
        }
        return personRecord;
    }
}
