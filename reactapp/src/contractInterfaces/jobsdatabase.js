import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xd903587aDBe002d5853E306EeF2b2105154eDd2e';

export default new web3.eth.Contract(jobs, address);
