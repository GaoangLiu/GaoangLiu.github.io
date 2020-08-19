import re, os, sys
import bisect


class Blog:
    def _export_to_post(self, fpath):
        lo = self._get_layouts(fpath)
        lo['layout'] = 'post'
        nm = lo['date'] + '-' + lo['title'].replace(' ', '-') + '.md'
        year = lo['date'].split('-')[0]
        postpath = f'tmp/{year}/{nm}'
        if not os.path.exists(f'tmp/{year}'):
            os.makedirs(f'tmp/{year}')

        head = "---\n"
        for k, v in lo.items():
            head += k + ": "
            # Some theme divide tags by ','
            head += " ".join(map(lambda e: e.lower().replace(' ', '_'),
                                 v)) if type(v) == list else v
            head += "\n"
        head += "author: GaoangLau\n---"
        print(postpath, head)
        con = self._get_contents(fpath)
        open(postpath, 'w').write(head + con)

    def _get_contents(self, fpath):
        f = open(fpath, 'r').read()
        cs = re.findall(r'---.*?---(.*)', f, re.DOTALL)[0]
        if "{:toc}" not in cs:
            i = 0
            while cs[i] != '\n' or i < 30:
                i += 1
            cs = cs[:i] + '\n\n\n' + cs[i:]
            return "\n* content\n{:toc}\n" + cs
        return cs

    def _get_layouts(self, fpath):
        # Get layout info (title, tag, categorical, etc)
        layouts = dict()
        with open(fpath, 'r') as f:
            header = re.findall(r'---(.*?)---', f.read(), re.DOTALL)[0]
            print(header)
            for line in header.split('\n'):
                if ':' not in line: continue
                key, value = line.split(':')
                value = re.sub('[\[\]]', '', value).strip()
                layouts.update((k, value) for k in ('layout', 'title', 'date')
                               if k in key)

                # Parse out tags and categories
                if 'tags' in key:
                    layouts['tags'] = [t.strip() for t in value.split(',')]

            cats = re.findall(r'cat[eo]g[oe]ries:(.*)', header, re.DOTALL)
            if cats:
                cats = list(
                    filter(
                        lambda e: e,
                        map(lambda e: e.strip('-').strip(),
                            cats[0].split('\n'))))
            layouts['categories'] = cats
        print(layouts)
        return layouts

    def post_drafts(self):
        # os.remove('_posts/*')
        draft_dir = '_drafts'
        for f in os.listdir('_drafts'):
            full_path = f'{draft_dir}/{f}'
            if os.path.isfile(full_path) and full_path.endswith('.md'):
                print(full_path)
                self._export_to_post(full_path)
                # return


if __name__ == "__main__":
    b = Blog()
    b.post_drafts()
