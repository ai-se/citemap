package example;

import java.util.HashMap;
import java.util.Map;

/*
 * Created on 20.11.2008
 */

public class Publication {
    private static Map<String, Publication> publicationMap = new HashMap<String, Publication>();

    private String key;

    private Publication(String key) {
        this.key = key;
        publicationMap.put(key, this);
    }


    static public Publication create(String key) {
        Publication p;
        p = searchPublication(key);
        if (p == null)
            p = new Publication(key);
        return p;
    }

    static public Publication searchPublication(String key) {
        return publicationMap.get(key);
    }
}